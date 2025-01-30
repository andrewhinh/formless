"""ETL for data to train classifiers and VLMs."""

import base64
import json
import logging
import math
import multiprocessing
import os
import random
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from xml.etree import ElementTree

import cairo
import modal
import numpy as np
import torch
from datasketch import MinHash
from dotenv import load_dotenv
from huggingface_hub import login
from imagehash import phash
from more_itertools import chunked
from PIL import Image
from pydantic import Field
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import coalesce, col, lit, udf
from pyspark.sql.types import ArrayType, IntegerType, StringType
from timm.data import create_transform, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import set_jit_fuser, setup_default_logging
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from utils import (
    DATA_VOLUME,
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    GPU_IMAGE,
    IN_PROD,
    MINUTES,
    NAME,
    PARENT_PATH,
    SECRETS,
    VOLUME_CONFIG,
)

# -----------------------------------------------------------------------------

# classifier training data config

RANDOM_SEED = 42
SPLITS = ["train", "valid", "test"]
N_SAMPLES_PER_SPLIT_CLS = {
    "train": 1000.0,
    "valid": 100.0,
    "test": 100.0,
}
QUALITIES = ["1", "2", "3"]

SLOW_RATER = "Qwen/Qwen2-VL-2B-Instruct-AWQ"  # pretrained model or ckpt
SLOW_RATER_TOKENIZER = "Qwen/Qwen2-VL-2B-Instruct-AWQ"  # pretrained tokenizer
QUANTIZATION = "awq_marlin"
KV_CACHE_DTYPE = None  # "fp8_e5m2"
ENFORCE_EAGER = False
MAX_NUM_SEQS = 1
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TEMPERATURE = 0.2
TOP_P = 0.001
REPEATION_PENALTY = 1.05
MAX_TOKENS = 5
STOP_TOKEN_IDS = []
PROMPT = """
You are given an image of a math expression and its corresponding annotation.
Determine the quality of the handwriting in the image using the additive 3-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
1) Add 1 point if the writing is very difficult to read.
2) Add another point if the writing is partially unreadable.
3) Add a third point if the writing is completely clear and legible.
Return the quality of the handwriting as a number between 1 and 3.
"""

# -----------------------------------------------------------------------------

# sft training data config

FAST_RATER = "hf_hub:andrewhinh/resnet152-cls"
BS = 2048  # max on RTX 3090

## imports

HAS_NATIVE_AMP = False
try:
    if torch.cuda.amp.autocast is not None:
        HAS_NATIVE_AMP = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion  # noqa: F401

    HAS_FUNCTORCH = True
except ImportError:
    HAS_FUNCTORCH = False

CHANNELS_LAST = False  # Use channels_last memory layout
FUSER = ""  # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')

## scripting / codegen
TORCHSCRIPT = False  # torch.jit.script the full model
AOT_AUTOGRAD = False  # Enable AOT Autograd support.

## device & distributed
if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 1
DEVICE = torch.device("cuda" if GPU_COUNT > 0 else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = True  # use Native AMP for mixed precision training
AMP_DTYPE = "bfloat16"  # lower precision AMP dtype (default: float16)
HAS_COMPILE = hasattr(torch, "compile")
TORCH_COMPILE = "inductor"  # Enable compilation w/ specified backend (default: inductor).

## misc
TEST_POOL = False  # enable test time pool
TOPK = 1  # Top-k

## dedup
NUM_PERM = 64  # larger = high acc but high mem usage
HASH_SZ = 8  # larger = more accurate but slower
THRESHOLD = 0.2  # larger = less duplicates
N_SAMPLES_PER_SPLIT_SFT = {
    "train": 10000.0,
    "valid": 1000.0,
    "test": 1000.0,
}

# -----------------------------------------------------------------------------

# setup
logging.getLogger("timm").setLevel(
    logging.WARNING
)  # disable "Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.""
spark = (
    SparkSession.builder.config("spark.executor.memory", "20G")
    .config("spark.driver.memory", "20G")
    .config("spark.driver.maxResultSize", "1G")
    .config("spark.sql.shuffle.partitions", "300")
    .config("spark.worker.cleanup.enabled", "true")
    .config("spark.worker.cleanup.interval", "1800")
    .config("spark.worker.cleanup.appDataTtl", "86400")
    .config("spark.local.dir", "/tmp/spark-temp")  # noqa: S108
    .config("spark.dynamicAllocation.enabled", "true")
    .config("spark.dynamicAllocation.minExecutors", "1")
    .config("spark.dynamicAllocation.maxExecutors", "10")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate()
)
random.seed(RANDOM_SEED)
load_dotenv(".env" if IN_PROD else ".env.dev")


## container startup fn
def download_models():
    from huggingface_hub import snapshot_download

    login(token=os.getenv("HF_TOKEN"), new_session=False)

    for model in [SLOW_RATER, SLOW_RATER_TOKENIZER, FAST_RATER.split("hf_hub:")[1]]:
        if not os.path.exists(model):
            snapshot_download(
                model,
                ignore_patterns=["*.pt", "*.bin"],
            )
        else:  # check if preprocessor_config.json was successfully copied; if not, do so
            if not os.path.exists(f"{model}/preprocessor_config.json"):
                tok_path = snapshot_download(
                    model,
                    ignore_patterns=["*.pt", "*.bin"],
                )
                os.rename(f"{tok_path}/preprocessor_config.json", f"{model}/preprocessor_config.json")


def setup_classifier(model_name: str):  # noqa: C901
    setup_default_logging()

    if GPU_COUNT > 0:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if AMP:
        assert HAS_NATIVE_AMP, "Please update PyTorch to a version with native AMP (or use APEX)."
        assert AMP_DTYPE in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=DEVICE.type, dtype=amp_dtype)

    if FUSER:
        set_jit_fuser(FUSER)

    # create model
    model = create_model(model_name, pretrained=True)
    data_config = resolve_data_config(
        {
            "model": model_name,
        },
        model=model,
    )
    transforms = create_transform(**data_config, is_training=False)
    if TEST_POOL:
        model, _ = apply_test_time_pool(model, data_config)

    model = model.to(DEVICE)
    model.eval()
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    if TORCHSCRIPT:
        model = torch.jit.script(model)
    elif TORCH_COMPILE:
        assert HAS_COMPILE, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        torch._dynamo.reset()
        model = torch.compile(model, backend=TORCH_COMPILE)
    elif AOT_AUTOGRAD:
        assert HAS_FUNCTORCH, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if GPU_COUNT > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(GPU_COUNT)))

    return transforms, amp_autocast, model


# -----------------------------------------------------------------------------

# Modal
IMAGE = (
    GPU_IMAGE.apt_install(
        [
            "libcairo2-dev",  # required to build pycairo
            "libjpeg-dev",  # required to build pycairo
            "libgif-dev",  # required to build pycairo
            "openjdk-11-jdk",  # required to build pyspark
        ]
    )
    .pip_install(
        "pycairo==1.27.0",
        "pydantic==2.10.4",
        "tqdm==4.67.1",
        "datasketch==1.6.5",
        "ImageHash==4.3.1",
        "pyspark==3.5.4",
    )
    .run_function(
        download_models,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
)
ETL_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "l4"
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-etl"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------

# helper cls/fns


@dataclass
class Ink:
    # Every stroke in the ink.
    # Each stroke array has shape (3, number of points), where the first
    # dimensions are (x, y, timestamp), in that order.
    strokes: list[np.ndarray] = Field(
        ...,
        description="Every stroke in the ink. Each stroke array has shape (3, number of points), where the first dimensions are (x, y, timestamp), in that order.",
    )
    # Metadata present in the InkML.
    annotations: dict[str, str] = Field(
        ...,
        description="Metadata present in the InkML.",
    )


def render_ink(
    ink: Ink,
    *,
    margin: int = 10,
    stroke_width: float = 1.5,
    stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Image:
    """Renders an ink as a PIL image using Cairo.

    The image size is chosen to fit the entire ink while having one pixel per
    InkML unit.

    Args:
    margin: size of the blank margin around the image (pixels)
    stroke_width: width of each stroke (pixels)
    stroke_color: color to paint the strokes with
    background_color: color to fill the background with

    Returns
    -------
    Rendered ink, as a PIL image.
    """
    # Compute transformation to fit the ink in the image.
    xmin, ymin = np.vstack([stroke[:2].min(axis=1) for stroke in ink.strokes]).min(axis=0)
    xmax, ymax = np.vstack([stroke[:2].max(axis=1) for stroke in ink.strokes]).max(axis=0)
    width = int(xmax - xmin + 2 * margin)
    height = int(ymax - ymin + 2 * margin)

    shift_x = -xmin + margin
    shift_y = -ymin + margin

    def apply_transform(ink_x: float, ink_y: float):
        return ink_x + shift_x, ink_y + shift_y

    # Create the canvas with the background color
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*background_color)
    ctx.paint()

    # Set pen parameters
    ctx.set_source_rgb(*stroke_color)
    ctx.set_line_width(stroke_width)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)

    for stroke in ink.strokes:
        if len(stroke[0]) == 1:
            # For isolated points we just draw a filled disk with a diameter equal
            # to the line width.
            x, y = apply_transform(stroke[0, 0], stroke[1, 0])
            ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
            ctx.fill()

        else:
            ctx.move_to(*apply_transform(stroke[0, 0], stroke[1, 0]))

            for ink_x, ink_y in stroke[:2, 1:].T:
                ctx.line_to(*apply_transform(ink_x, ink_y))
            ctx.stroke()

    # cairo to pil
    size = (surface.get_width(), surface.get_height())
    stride = surface.get_stride()
    with surface.get_data() as memory:
        return Image.frombuffer("RGB", size, memory.tobytes(), "raw", "BGRX", stride)


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def extract_ink_metadata(input_path: Path, save_path: Path) -> dict[Path, str]:
    """
    Extract ink metadata from inkml file and render ink as image.
    """
    # read inkml
    with open(input_path, "r") as f:
        root = ElementTree.fromstring(f.read())  # noqa: S314
    strokes = []
    annotations = {}
    for element in root:
        tag_name = element.tag.removeprefix("{http://www.w3.org/2003/InkML}")
        if tag_name == "annotation":
            annotations[element.attrib.get("type")] = element.text
        elif tag_name == "trace":
            points = element.text.split(",")
            stroke_x, stroke_y, stroke_t = [], [], []
            for point in points:
                x, y, t = point.split(" ")
                stroke_x.append(float(x))
                stroke_y.append(float(y))
                stroke_t.append(float(t))
            strokes.append(np.array((stroke_x, stroke_y, stroke_t)))
    ink = Ink(strokes=strokes, annotations=annotations)

    # render ink and save
    img = render_ink(ink)
    if save_path.exists():
        save_path.unlink()
    img.save(save_path)
    label = ink.annotations["label"]
    return {"img_path": save_path, "label": label}


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def analyze_ink(img_path: Path) -> int:
    """
    Analyze ink and return writing quality.
    """
    with open(img_path, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{base64_img}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    if img_url is not None:
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": img_url}})

    global llm
    global sampling_params
    # load pretrained vlm if not already loaded
    if "llm" not in globals():
        llm = LLM(
            model=SLOW_RATER,
            enforce_eager=ENFORCE_EAGER,
            max_num_seqs=MAX_NUM_SEQS,
            tensor_parallel_size=GPU_COUNT,
            trust_remote_code=True,
            mm_processor_kwargs={
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            },
            **{k: v for k, v in [("quantization", QUANTIZATION), ("kv_cache_dtype", KV_CACHE_DTYPE)] if v is not None},
        )
    if "sampling_params" not in globals():
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPEATION_PENALTY,
            max_tokens=MAX_TOKENS,
            stop_token_ids=STOP_TOKEN_IDS,
            guided_decoding=GuidedDecodingParams(choice=[1, 2, 3]),
        )
    outputs = llm.chat(messages, sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    writing_quality = int(generated_text)

    img = Image.open(img_path).convert("RGB")
    if not os.path.exists(img_path.parent / str(writing_quality)):
        os.mkdir(img_path.parent / str(writing_quality))
    img.save(img_path.parent / str(writing_quality) / img_path.name)
    return writing_quality


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def classify_ink(img_paths: list[Path]) -> list[int]:
    """
    Classify ink and return writing quality.
    """
    global transforms, amp_autocast, classifier
    if "classifier" not in globals():
        transforms, amp_autocast, classifier = setup_classifier(FAST_RATER)
        classifier.eval()

    img_pts = torch.cat(
        thread_map(
            lambda p: transforms(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE),
            img_paths,
        )
    )

    with torch.no_grad():
        with amp_autocast():
            outputs = classifier(img_pts)

    labels = classifier.pretrained_cfg["label_names"]
    predictions = outputs.softmax(-1).topk(TOPK, dim=-1)
    writing_qualities = [int(labels[idx[0]]) for idx in predictions.indices]

    for img_path, writing_quality in zip(img_paths, writing_qualities, strict=False):
        output_dir = img_path.parent / str(writing_quality)
        output_dir.mkdir(exist_ok=True)
        Image.open(img_path).convert("RGB").save(output_dir / img_path.name)

    return writing_qualities


def compute_minhash(phash_str):
    """
    Compute the minhash of a perceptual hash.

    Args:
    phash_str (str): The perceptual hash of the image.

    Returns
    -------
    list: The minhash of the image.
    """
    m = MinHash(num_perm=NUM_PERM)
    m.update(phash_str.encode("utf8"))
    return [int(hash_value) for hash_value in m.hashvalues]  # Convert uint64 to int


def generate_json(row):
    """
    Generate JSON entry for each row in the DataFrame.
    """
    return json.dumps(
        {
            "messages": [
                {
                    "content": DEFAULT_SYSTEM_PROMPT,
                    "role": "system",
                },
                {
                    "content": f"<image>{DEFAULT_QUESTION}",
                    "role": "user",
                },
                {
                    "content": row.label,
                    "role": "assistant",
                },
            ],
            "images": [
                row.img_path,
            ],
        }
    )


# -----------------------------------------------------------------------------

# main


def main(cls: bool, sft: bool, dpo: bool):  # noqa: C901
    if not cls and not sft and not dpo:
        raise ValueError("Must specify at least one of `cls`, `sft`, or `dpo`")

    # load df; o/w create and save for later use
    if modal.is_local():
        DATA_VOL_PATH = str(PARENT_PATH / "training" / "artifacts" / "mathwriting-2024")
        if not os.path.exists(DATA_VOL_PATH):
            raise Exception(f"""
    {DATA_VOL_PATH} does not exist.
    """)
    else:
        DATA_VOL_PATH = f"/{DATA_VOLUME}"
    PARQUET_FILENAME = f"{DATA_VOL_PATH}/data.parquet"

    df = spark.createDataFrame([], schema="img_path string, label string, split string")
    split_cts = {}
    if os.path.exists(PARQUET_FILENAME):
        print(f"Loading existing dataframe from {PARQUET_FILENAME}")
        df = spark.read.parquet(PARQUET_FILENAME)
        for split in SPLITS:
            split_cts[split] = df.filter(col("split") == split).count()
            print(f"Loaded {split_cts[split]} samples for split {split}")
    else:
        print(f"Creating new dataframe from {DATA_VOL_PATH}")
        for split in SPLITS:
            # extract ink metadata
            filenames = list(Path(f"{DATA_VOL_PATH}/{split}").glob("*.inkml"))
            save_paths = [Path(f"{DATA_VOL_PATH}/{split}/{filename.stem}.png") for filename in filenames]
            if modal.is_local():
                split_stats = list(
                    tqdm(
                        thread_map(
                            extract_ink_metadata.local,
                            filenames,
                            save_paths,
                            max_workers=multiprocessing.cpu_count(),
                        ),
                        desc=split,
                        total=len(filenames),
                    )
                )
            else:
                split_stats = extract_ink_metadata.starmap(
                    (filename, save_path) for filename, save_path in zip(filenames, save_paths, strict=True)
                )

            # write to df
            split_df = spark.createDataFrame(
                [
                    Row(
                        img_path=str(stat["img_path"]),
                        label=stat["label"],
                        split=split,
                    )
                    for stat in split_stats
                ],
                schema=df.schema,
            )
            split_cts[split] = split_df.count()
            new_entries = split_df.join(df, on=["img_path", "split"], how="left_anti")
            df = df.unionByName(new_entries.select(df.columns))
            print(f"Collected {split_cts[split]} samples for split {split}")

        df.write.mode("overwrite").parquet(PARQUET_FILENAME)

    if cls:
        # run model to assign writing quality to random subset
        df = df.withColumn("writing_quality", lit(None).cast(IntegerType()))
        all_updates = []
        for split in SPLITS:
            ## get random subset of split data
            split_df = df.filter(col("split") == split)
            split_filter_df = split_df.sample(
                False, N_SAMPLES_PER_SPLIT_CLS[split] / split_cts[split], float(RANDOM_SEED)
            )

            ## run model
            img_paths = [Path(row.img_path) for row in split_filter_df.select("img_path").collect()]
            if modal.is_local():
                writing_qualities = list(
                    tqdm(
                        (analyze_ink.local(p) for p in img_paths),
                        desc=split,
                        total=len(img_paths),
                    )
                )
            else:
                writing_qualities = analyze_ink.map(img_paths)

            ## save to write later
            new_schema = ["img_path", "split", "writing_quality"]
            data_for_split = [(str(p), split, wq) for p, wq in zip(img_paths, writing_qualities, strict=False)]
            updates_df = spark.createDataFrame(data_for_split, schema=new_schema)
            all_updates.append(updates_df)
            print(f"Labeled {N_SAMPLES_PER_SPLIT_CLS[split]} samples for split {split}")

        ## write to df
        final_updates_df = all_updates[0]
        for update_df in all_updates[1:]:
            final_updates_df = final_updates_df.unionByName(update_df)
        df = (
            df.alias("main")
            .join(final_updates_df.alias("upd"), on=["img_path", "split"], how="left")
            .select(
                col("main.img_path"),
                col("main.label"),
                col("main.split"),
                # If main already had writing_quality, we merge it; otherwise just pick upd
                coalesce(col("upd.writing_quality"), col("main.writing_quality")).alias("writing_quality"),
            )
        )
        df.write.mode("overwrite").parquet(PARQUET_FILENAME)
    if sft:
        # run trained classifier on df
        all_updates = []
        for split in SPLITS:
            ## get non-labeled samples
            split_df = df.filter(col("split") == split)
            split_df_wq_null = split_df.filter(split_df.writing_quality.isNull())

            ## run model
            img_paths = [Path(row.img_path) for row in split_df_wq_null.select("img_path").collect()]
            if not img_paths:
                continue
            if modal.is_local():
                img_batches = list(chunked(img_paths, BS))
                writing_qualities = list(
                    tqdm(
                        chain.from_iterable(classify_ink.local(p) for p in img_batches),
                        desc=split,
                        total=len(img_batches),
                    )
                )
            else:
                writing_qualities = classify_ink.map(img_paths)

            ## save to write later
            new_schema = ["img_path", "split", "writing_quality"]
            data_for_split = [(str(p), split, wq) for p, wq in zip(img_paths, writing_qualities, strict=False)]
            updates_df = spark.createDataFrame(data_for_split, schema=new_schema)
            all_updates.append(updates_df)

        ## write to df
        if all_updates:
            final_updates_df = all_updates[0]
            for update_df in all_updates[1:]:
                final_updates_df = final_updates_df.unionByName(update_df)
            df = (
                df.alias("main")
                .join(final_updates_df.alias("upd"), on=["img_path", "split"], how="left")
                .select(
                    col("main.img_path"),
                    col("main.label"),
                    col("main.split"),
                    coalesce(col("upd.writing_quality"), col("main.writing_quality")).alias("writing_quality"),
                )
            )
            df.write.mode("overwrite").parquet(PARQUET_FILENAME)

        # filter to only get data with writing quality == 1
        filtered_dfs = []
        for split in SPLITS:
            split_df = df.filter(col("split") == split)
            split_filter_df = split_df.filter(split_df.writing_quality == 1)
            filtered_dfs.append(split_filter_df)
        filter_df = filtered_dfs[0]
        for filtered_df in filtered_dfs[1:]:
            filter_df = filter_df.unionByName(filtered_df)

        # deduplication
        dedup_dfs = []

        ## compute minhash vec
        compute_phash_udf = udf(lambda img_path: str(phash(Image.open(img_path), hash_size=HASH_SZ)), StringType())
        compute_minhash_udf = udf(compute_minhash, ArrayType(IntegerType()))
        vector_udf = udf(lambda minhash: Vectors.dense(minhash), VectorUDT())

        filter_df = filter_df.withColumn("phash", compute_phash_udf(col("img_path")))
        filter_df = filter_df.withColumn("minhash", compute_minhash_udf(col("phash")))
        filter_df = filter_df.withColumn("minhash_vec", vector_udf(col("minhash")))

        ## dedup with lsh and randomly sample
        for split in SPLITS:
            split_df = filter_df.filter(col("split") == split)
            lsh = MinHashLSH(inputCol="minhash_vec", outputCol="hashes", numHashTables=NUM_PERM)
            lsh_model = lsh.fit(split_df)
            deduped_df = (
                lsh_model.approxSimilarityJoin(split_df, split_df, THRESHOLD, distCol="distance")
                .filter("datasetA.img_path != datasetB.img_path")
                .select("datasetA.*")
                .distinct()
            )

            split_dedup_df = deduped_df.drop("phash", "minhash", "minhash_vec", "hashes")
            split_dedup_df = split_dedup_df.sample(
                False, N_SAMPLES_PER_SPLIT_SFT[split] / split_cts[split], float(RANDOM_SEED)
            )
            dedup_dfs.append(split_dedup_df)
            print(f"Collected {N_SAMPLES_PER_SPLIT_SFT[split]} samples for split {split}")
        final_dedup_df = dedup_dfs[0]
        for dedup_df in dedup_dfs[1:]:
            final_dedup_df = final_dedup_df.unionByName(dedup_df)

        # write to json
        for split in SPLITS:
            split_df = df.filter(col("split") == split)

            data = split_df.select("label", "img_path").collect()
            labels = [row.label for row in data]
            img_paths = [row.img_path for row in data]
            json_output = [
                {
                    "messages": [
                        {
                            "content": DEFAULT_SYSTEM_PROMPT,
                            "role": "system",
                        },
                        {
                            "content": f"<image>{DEFAULT_QUESTION}",
                            "role": "user",
                        },
                        {
                            "content": label,
                            "role": "assistant",
                        },
                    ],
                    "images": [
                        img_path,
                    ],
                }
                for label, img_path in zip(
                    labels,
                    img_paths,
                    strict=False,
                )
            ]

            output_path = Path(f"{DATA_VOL_PATH}/{split}/data.json")
            with open(output_path, "w") as f:
                json.dump(json_output, f, indent=4)
            print(f"Wrote to {output_path}")
    if dpo:
        # load df

        # run trained VLM on val data

        # identify subset of val samples with worst perf

        # prompt user to manually label data

        # write to json
        pass


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def run(cls: bool, sft: bool, dpo: bool):
    main(cls, sft, dpo)


@app.local_entrypoint()
def local(cls: bool = False, sft: bool = False, dpo: bool = False):
    run.remote(cls, sft, dpo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    args = parser.parse_args()
    download_models()
    main(args.cls, args.sft, args.dpo)


# TODO:
# - get better data (cutmix/mixup, filters, etc.)
