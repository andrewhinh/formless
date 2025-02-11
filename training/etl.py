"""ETL for data to train classifiers and VLMs."""

import base64
import hashlib
import json
import logging
import math
import multiprocessing
import os
import random
from collections import Counter
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
import yaml
from datasketch import MinHash
from dotenv import load_dotenv
from huggingface_hub import login
from imagehash import phash
from more_itertools import chunked
from PIL import Image, ImageFile, ImageFilter
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
    APP_NAME,
    DATA_VOLUME,
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    GPU_IMAGE,
    HF_USERNAME,
    IN_PROD,
    MINUTES,
    PARENT_PATH,
    SECRETS,
    VOLUME_CONFIG,
)

# setup
ImageFile.LOAD_TRUNCATED_IMAGES = True
TABLE_BS = 65536  # so no memory overload when pulling data from df
RANDOM_SEED = 42
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
    .config("spark.sql.autoBroadcastJoinThreshold", -1)
    .getOrCreate()
)
random.seed(RANDOM_SEED)
load_dotenv(".env" if IN_PROD else ".env.dev")

# -----------------------------------------------------------------------------

# vlm config

QUANTIZATION = "awq_marlin"  # "awq_marlin"
KV_CACHE_DTYPE = None  # "fp8_e5m2"
ENFORCE_EAGER = False
MAX_NUM_SEQS = 16  # max for 3090
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TEMPERATURE = 0.0
TOP_P = 0.001
REPEATION_PENALTY = 1.05
STOP_TOKEN_IDS = []

MAX_SCORE_TOKENS = 3
MAX_VLM_TOKENS = 2048

# LATEX_GRAMMER = """
# start: math_expr

# math_expr: inline_math | display_math

# inline_math: "$" expr "$"
# display_math: BEGIN "{" ENV "}" expr END "{" ENV "}"

# ?expr: term
#      | expr operator term   -> binop

# ?term: NUMBER
#      | VARIABLE
#      | function
#      | "{" expr "}"
#      | FRAC "{" expr "}" "{" expr "}"

# function: SIN "(" expr ")"
#         | COS "(" expr ")"
#         | TAN "(" expr ")"
#         | LOG "(" expr ")"
#         | EXP "(" expr ")"
#         | SQRT "{" expr "}"

# operator: "+" | "-" | "*" | "/" | "=" | "^"

# BEGIN: "\\begin"
# END: "\\end"
# FRAC: "\\frac"
# SIN: "\\sin"
# COS: "\\cos"
# TAN: "\\tan"
# LOG: "\\log"
# EXP: "\\exp"
# SQRT: "\\sqrt"

# ENV: /[a-zA-Z]+/

# %import common.NUMBER
# %import common.CNAME -> VARIABLE
# %import common.WS
# %ignore WS
# """


# -----------------------------------------------------------------------------

# classifier training data config

SPLITS = ["train", "valid", "test"]
N_SAMPLES_PER_SPLIT_CLS = {
    "train": 800.0,
    "valid": 100.0,
    "test": 100.0,
}
CLASSES = ["1", "2", "3"]

SLOW_RATER = "Qwen/Qwen2-VL-7B-Instruct-AWQ"  # pretrained model or ckpt
SLOW_RATER_TOKENIZER = "Qwen/Qwen2-VL-7B-Instruct-AWQ"  # pretrained tokenizer
DIFFICULTY_PROMPT = """
You are given an image of a handwritten math expression and its corresponding annotation.
Determine the grade level of the expression in the image using the additive 3-point scoring system described below.
Points are accumulated based on the satisfaction of each criterion:
1) Add 1 point if the expression is something that an elementary to middle school student could understand.
2) Add another point if the expression is something that a high school student could understand.
3) Add a third point if the expression is something that an undergrad to grad student could understand.
Return the difficulty of the expression as a number between 1 and 3 where
1 is the easiest and 3 is the most difficult.
"""

# -----------------------------------------------------------------------------

# sft training data config

FAST_RATER = f"hf_hub:{HF_USERNAME}/{APP_NAME}-resnet152-difficulty"
CLS_RUN_BS = 2048  # max on RTX 3090

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
    "train": 800,
    "valid": 100,
    "test": 100,
}  # only train data will be written to json, valid/test will be used for eval

## stitch
STITCH_BS_MAX = 128  # 128 ~= 2048 tokens
ROTATE_MAX = 10  # degrees

## json paths
if modal.is_local():
    DATA_VOL_PATH = str(PARENT_PATH / "training" / "artifacts" / "mathwriting-2024")
    if not os.path.exists(DATA_VOL_PATH):
        raise Exception(f"""
{DATA_VOL_PATH} does not exist.
""")
else:
    DATA_VOL_PATH = f"/{DATA_VOLUME}"
SFT_TRAIN_JSON = Path(f"{DATA_VOL_PATH}/sft_train.json")
SFT_VAL_JSON = Path(f"{DATA_VOL_PATH}/sft_valid.json")
SFT_TEST_JSON = Path(f"{DATA_VOL_PATH}/sft_test.json")

# -----------------------------------------------------------------------------

# dpo training data config

SLOW_RUNNER = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-sft-merged-awq"  # pretrained model or ckpt
SLOW_RUNNER_TOKENIZER = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-sft-merged-awq"  # pretrained tokenizer
DPO_TRAIN_JSON = Path(f"{DATA_VOL_PATH}/dpo_train.json")

# -----------------------------------------------------------------------------


## container startup fn
def download_models():
    from huggingface_hub import snapshot_download

    login(token=os.getenv("HF_TOKEN"), new_session=False)

    for model in [SLOW_RATER, SLOW_RATER_TOKENIZER, FAST_RATER.split("hf_hub:")[1], SLOW_RUNNER, SLOW_RUNNER_TOKENIZER]:
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
TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "l4"
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

app = modal.App(name=f"{APP_NAME}-etl")

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
    timeout=TIMEOUT,
)
def extract_ink_metadata(input_path: Path) -> dict[Path, str]:
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
    save_path = input_path.parent / (input_path.stem + ".png")
    if save_path.exists():
        save_path.unlink()
    img.save(save_path)
    label = ink.annotations["label"]
    return {"img_path": save_path, "label": label}


def run_model(
    img_paths: list[Path], model: str, prompt: str, max_tokens: int = None, guided_decoding: GuidedDecodingParams = None
) -> list[str]:
    """
    Run model on image(s) with prompt.
    """
    conversations = []
    for img_path in img_paths:
        with open(img_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{base64_img}"
        conversations.append(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                },
            ]
        )

    global llm
    global sampling_params
    # load pretrained vlm if not already loaded
    if "llm" not in globals():
        llm = LLM(
            model=model,
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
            max_tokens=max_tokens,
            stop_token_ids=STOP_TOKEN_IDS,
            guided_decoding=guided_decoding,
        )
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    preds = [out.outputs[0].text.strip() for out in outputs]
    return preds


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def analyze_inks(img_paths: list[Path]) -> list[int]:
    """
    Analyze ink(s) and return score(s).
    """
    preds = run_model(
        SLOW_RATER,
        DIFFICULTY_PROMPT,
        MAX_SCORE_TOKENS,
        GuidedDecodingParams(choice=[int(cls) for cls in CLASSES]),
        img_paths,
    )
    scores = [int(pred) for pred in preds]
    for img_path, score in zip(img_paths, scores, strict=True):
        img = Image.open(img_path).convert("RGB")
        if not os.path.exists(img_path.parent / str(score)):
            os.mkdir(img_path.parent / str(score))
        img.save(img_path.parent / str(score) / img_path.name)
    return scores


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def classify_ink(img_paths: list[Path]) -> list[int]:
    """
    Classify ink and return score.
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

    cls_names = classifier.pretrained_cfg["label_names"]
    predictions = outputs.softmax(-1).topk(TOPK, dim=-1)
    scores = [int(cls_names[idx[0]]) for idx in predictions.indices]
    return scores


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


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def stitch_imgs(img_paths: list[Path]) -> Path:
    """
    Given a list of image file paths, stitch them together into a grid
    so that the final image is the same size as one of the input images.

    Each image is resized to fit into its cell.
    """
    images = [Image.open(path).convert("RGBA") for path in img_paths]
    w, h = images[0].size

    # Ensure final image dimensions are within the specified range
    min_dim = 3 * MIN_PIXELS
    max_dim = 3 * MAX_PIXELS
    w = max(min_dim, min(w, max_dim))
    h = max(min_dim, min(h, max_dim))

    num_images = len(images)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    tile_width = w // grid_cols
    tile_height = h // grid_rows

    ## use RGBA to handle transparency (default: white)
    stitched_image = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    for i, img in enumerate(images):
        img_resized = img.resize((tile_width, tile_height), Image.LANCZOS)

        ### rotate
        angle = random.uniform(-ROTATE_MAX, ROTATE_MAX)
        img_rotated = img_resized.rotate(angle, resample=Image.BICUBIC, expand=True)

        ### gaussian blur
        img_blurred = img_rotated.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

        ### rotation with expand=True changes the size
        ### -> calculate pos to center transformed image
        ### within its grid cell, then add rand offset.
        w_trans, h_trans = img_blurred.size
        base_offset_x = (tile_width - w_trans) // 2
        base_offset_y = (tile_height - h_trans) // 2
        rand_offset_x = random.randint(-tile_width // 8, tile_width // 8)
        rand_offset_y = random.randint(-tile_height // 8, tile_height // 8)
        offset_x = base_offset_x + rand_offset_x
        offset_y = base_offset_y + rand_offset_y

        col = i % grid_cols
        row = i // grid_cols
        cell_x = col * tile_width
        cell_y = row * tile_height
        paste_x = cell_x + offset_x
        paste_y = cell_y + offset_y

        ### paste with alpha channel (= mask) to handle transparency
        stitched_image.paste(img_blurred, (paste_x, paste_y), img_blurred)

    unique_string = "".join(sorted([str(path) for path in img_paths]))
    hash_digest = hashlib.md5(unique_string.encode("utf-8")).hexdigest()  # noqa: S324
    save_path = img_paths[0].parent / f"{hash_digest}.png"
    final_image = stitched_image.convert("RGB")
    final_image.save(save_path)
    return save_path


def write_sft_json(json_path: Path, img_paths: list, labels: list):
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>{DEFAULT_QUESTION}",
                        },
                        {
                            "from": "gpt",
                            "value": label,
                        },
                    ],
                    "images": [
                        str(img_path),
                    ],
                }
                for img_path, label in zip(
                    img_paths,
                    labels,
                    strict=True,
                )
            ],
            f,
            indent=4,
        )


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def ft_pred_ink(img_paths: list[Path]) -> list[str]:
    """
    Run trained VLM on ink(s).
    """
    return run_model(
        img_paths,
        SLOW_RUNNER,
        DEFAULT_QUESTION,
        MAX_VLM_TOKENS,
        # GuidedDecodingParams(grammar=LATEX_GRAMMER, backend="outlines"),
    )


def write_dpo_json(json_path: Path, img_paths: list, preds: list, labels: list):
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>{DEFAULT_QUESTION}",
                        },
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": label,
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": pred,
                    },
                    "images": [
                        str(img_path),
                    ],
                }
                for img_path, pred, label in zip(
                    img_paths,
                    preds,
                    labels,
                    strict=True,
                )
            ],
            f,
            indent=4,
        )


# -----------------------------------------------------------------------------

# main


def main(cls: bool, sft: bool, dpo: bool):  # noqa: C901
    if not cls and not sft and not dpo:
        raise ValueError("Must specify at least one of `cls`, `sft`, or `dpo`")

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
            ink_paths = list(Path(f"{DATA_VOL_PATH}/{split}").glob("*.inkml"))
            if modal.is_local():
                split_stats = list(
                    tqdm(
                        thread_map(
                            extract_ink_metadata.local,
                            ink_paths,
                            max_workers=multiprocessing.cpu_count(),
                        ),
                        desc=split,
                        total=len(ink_paths),
                    )
                )
            else:
                split_stats = list(extract_ink_metadata.map(ink_paths))

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
        # run VLM to assign score to random subset
        df = df.withColumn("score", lit(None).cast(IntegerType()))
        df_parts = []
        for split in SPLITS:
            ## get random subset of split data
            split_df = df.filter(col("split") == split)
            split_df = split_df.sample(False, N_SAMPLES_PER_SPLIT_CLS[split] / split_cts[split], float(RANDOM_SEED))

            ## run
            img_paths = []
            for chunk in chunked(split_df.select("img_path").toLocalIterator(), TABLE_BS):
                img_paths.extend(Path(row.img_path) for row in chunk)
            if not img_paths:
                continue

            img_batches = list(chunked(img_paths, MAX_NUM_SEQS))
            if modal.is_local():
                scores = list(
                    tqdm(
                        chain.from_iterable(analyze_inks.local(batch) for batch in img_batches),
                        desc=split,
                        total=len(img_batches),
                    )
                )
            else:
                scores = list(analyze_inks.map(img_batches))

            ## save to write later
            split_df = spark.createDataFrame(
                [(str(p), split, sc) for p, sc in zip(img_paths, scores, strict=True)],
                schema=["img_path", "split", "score"],
            )
            df_parts.append(split_df)
            print(f"Labeled {N_SAMPLES_PER_SPLIT_CLS[split]} samples for split {split}")
            print(f"Score distribution: {Counter(scores)}")

        ## write to df
        split_df = df_parts[0]
        for update_df in df_parts[1:]:
            split_df = split_df.unionByName(update_df)
        df = (
            df.alias("main")
            .join(split_df.alias("upd"), on=["img_path", "split"], how="left")
            .select(
                col("main.img_path"),
                col("main.label"),
                col("main.split"),
                # If main already had score, we merge it; otherwise just pick upd
                coalesce(col("upd.score"), col("main.score")).alias("score"),
            )
        )
        df.write.mode("overwrite").parquet(PARQUET_FILENAME)
    if sft:
        # run trained classifier on df
        df_parts = []
        for split in SPLITS:
            ## get non-labeled samples
            split_df = df.filter(col("split") == split)
            split_df = split_df.filter(split_df.score.isNull())

            ## run model
            img_paths = []
            for chunk in chunked(split_df.select("img_path").toLocalIterator(), TABLE_BS):
                img_paths.extend(Path(row.img_path) for row in chunk)
            if not img_paths:
                continue

            img_batches = list(chunked(img_paths, CLS_RUN_BS))
            if modal.is_local():
                scores = list(
                    tqdm(
                        chain.from_iterable(classify_ink.local(p) for p in img_batches),
                        desc=split,
                        total=len(img_batches),
                    )
                )
            else:
                lst_scores = list(classify_ink.map(img_batches))
            scores = [item for lst in lst_scores for item in lst]

            ## save to write later
            split_df = spark.createDataFrame(
                [(str(p), split, sc) for p, sc in zip(img_paths, scores, strict=True)],
                schema=["img_path", "split", "score"],
            )
            df_parts.append(split_df)

        ## write to df
        if df_parts:
            split_df = df_parts[0]
            for update_df in df_parts[1:]:
                split_df = split_df.unionByName(update_df)
            df = (
                df.alias("main")
                .join(split_df.alias("upd"), on=["img_path", "split"], how="left")
                .select(
                    col("main.img_path"),
                    col("main.label"),
                    col("main.split"),
                    coalesce(col("upd.score"), col("main.score")).alias("score"),
                )
            )
            df.write.mode("overwrite").parquet(PARQUET_FILENAME)

        # deduplication
        df_parts = []

        ## compute minhash vec
        compute_phash_udf = udf(lambda img_path: str(phash(Image.open(img_path), hash_size=HASH_SZ)), StringType())
        compute_minhash_udf = udf(compute_minhash, ArrayType(IntegerType()))
        vector_udf = udf(lambda minhash: Vectors.dense(minhash), VectorUDT())
        df = (
            df.withColumn("phash", compute_phash_udf(col("img_path")))
            .withColumn("minhash", compute_minhash_udf(col("phash")))
            .withColumn("minhash_vec", vector_udf(col("minhash")))
        )

        ## dedup with lsh
        for split in SPLITS:
            split_df = df.filter(col("split") == split)

            ## lsh
            lsh = MinHashLSH(inputCol="minhash_vec", outputCol="hashes", numHashTables=NUM_PERM)
            lsh_model = lsh.fit(split_df)

            ## approx sim join
            split_df = (
                lsh_model.approxSimilarityJoin(split_df, split_df, THRESHOLD, distCol="distance")
                .filter("datasetA.img_path != datasetB.img_path")
                .select("datasetA.*")
                .distinct()
            )
            split_df = split_df.drop("phash", "minhash", "minhash_vec", "hashes")
            df_parts.append(split_df)
        split_df = df_parts[0]
        for update_df in df_parts[1:]:
            split_df = split_df.unionByName(update_df)

        # randomly stitch samples together
        img_paths, labels = (
            {split: [] for split in SPLITS},
            {split: [] for split in SPLITS},
        )
        for split in SPLITS:
            split_df = df.filter(col("split") == split)

            for score in CLASSES:
                split_filter_df = split_df.filter(split_df.score == score)

                paths, lbls = [], []
                for chunk in chunked(split_filter_df.select("img_path", "label").toLocalIterator(), TABLE_BS):
                    paths.extend([Path(row.img_path) for row in chunk])
                    lbls.extend([row.label for row in chunk])
                if not paths or not lbls:
                    continue

                ## rand sample
                path_batches = []
                combined_labels = []
                for _ in range(N_SAMPLES_PER_SPLIT_SFT[split] // len(CLASSES)):
                    num_sample = random.randint(1, STITCH_BS_MAX)
                    idxs = random.sample(range(len(paths)), min(num_sample, len(paths)))  # in case not enough
                    path_batches.append([paths[i] for i in idxs])
                    combined_labels.append("".join([lbls[i] for i in idxs]))

                ## stitch
                if modal.is_local():
                    stitched_img_paths = list(
                        tqdm(
                            thread_map(
                                stitch_imgs.local,
                                path_batches,
                                max_workers=multiprocessing.cpu_count(),
                                desc=split,
                            ),
                            total=len(path_batches),
                        )
                    )
                else:
                    stitched_img_paths = list(stitch_imgs.map(path_batches))

                img_paths[split].extend(stitched_img_paths)
                labels[split].extend(combined_labels)

            print(f"Generated {N_SAMPLES_PER_SPLIT_SFT[split]} samples for {split} split")

        # write to json
        write_sft_json(SFT_TRAIN_JSON, img_paths["train"], labels["train"])
        write_sft_json(SFT_VAL_JSON, img_paths["valid"], labels["valid"])
        write_sft_json(SFT_TEST_JSON, img_paths["test"], labels["test"])
    if dpo:
        # run model to determine which train samples it fails on
        for split in SPLITS:
            json_path = SFT_TRAIN_JSON if split == "train" else SFT_VAL_JSON if split == "valid" else SFT_TEST_JSON
            with open(json_path, "r") as f:
                read_ds = yaml.safe_load(f)
            img_paths = [sample["images"][0] for sample in read_ds]
            labels = [sample["conversations"][1]["value"] for sample in read_ds]

            ## run
            img_batches = list(chunked(img_paths, MAX_NUM_SEQS))
            if modal.is_local():
                preds = list(
                    tqdm(
                        chain.from_iterable(ft_pred_ink.local(batch) for batch in img_batches),
                        desc=split,
                        total=len(img_batches),
                    )
                )
            else:
                lst_preds = list(ft_pred_ink.map(img_batches))
                preds = [item for lst in lst_preds for item in lst]

            if split == "train":
                img_paths, labels, preds = zip(
                    *[
                        (path, label, pred)
                        for path, label, pred in zip(img_paths, labels, preds, strict=False)
                        if label != pred
                    ],
                    strict=False,
                )
                print(f"Generated {len(img_paths)} DPO samples for {split} split")
                write_dpo_json(DPO_TRAIN_JSON, img_paths, preds, labels)


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
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
