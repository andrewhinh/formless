"""ETL for data to train classifiers and VLMs."""

import base64
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import modal

from utils import DATA_VOLUME, DEFAULT_QUESTION, DEFAULT_SYSTEM_PROMPT, GPU_IMAGE, MINUTES, NAME, SECRETS, VOLUME_CONFIG

# -----------------------------------------------------------------------------

MODEL = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
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

QUALITIES = ["1", "2", "3"]

SPLITS = ["train"]
RANDOM_SEED = 42
N_SAMPLES = 1000
THRESHOLD = 0.8
NUM_PERM = 128
HASH_SZ = 16

# -----------------------------------------------------------------------------


# container build-time fns
def download_model():
    from huggingface_hub import login, snapshot_download

    login(token=os.getenv("HF_TOKEN"), new_session=False)
    snapshot_download(
        MODEL,
        ignore_patterns=["*.pt", "*.bin"],
    )


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
        download_model,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
)
ETL_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "l4"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-etl"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------

with IMAGE.imports():
    import cairo
    import numpy as np
    from datasketch import MinHash, MinHashLSH
    from imagehash import phash
    from PIL import Image
    from pydantic import Field
    from pyspark.sql import Row, SparkSession
    from pyspark.sql.functions import col
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
    random.seed(RANDOM_SEED)

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

    def dedup_per_split(split_df):
        lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)

        def is_duplicate(row):
            img_hash = phash(Image.open(row.img_path), hash_size=HASH_SZ)
            m = MinHash(num_perm=NUM_PERM)
            m.update(str(img_hash).encode("utf8"))
            if not lsh.query(m):
                lsh.insert(f"{row.img_path}_{row.writing_quality}", m)
                return False
            return True

        return split_df.rdd.filter(lambda row: not is_duplicate(row)).toDF(split_df.schema)


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def analyze_ink(filename: Path) -> dict[Path, int, str]:
    llm = LLM(
        model=MODEL,
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

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPEATION_PENALTY,
        max_tokens=MAX_TOKENS,
        stop_token_ids=STOP_TOKEN_IDS,
        guided_decoding=GuidedDecodingParams(choice=[1, 2, 3]),
    )

    def model_call(prompt: str, img_url: str = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        if img_url is not None:
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": img_url}})
        outputs = llm.chat(messages, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text

    # read inkml
    with open(filename, "r") as f:
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
    img_path = filename.with_suffix(".png")
    img = render_ink(ink)
    if os.path.exists(img_path):
        os.remove(img_path)
    img.save(img_path)
    with open(img_path, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{base64_img}"
    writing_quality = int(model_call(PROMPT, img_url))
    label = ink.annotations["label"]
    return {"img_path": img_path, "writing_quality": writing_quality, "label": label}


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def run(cls: bool, sft: bool, dpo: bool):  # noqa: C901
    if cls:
        # run pretrained vlm to assign writing quality to random subset
        df = spark.createDataFrame([], schema="img_path string, writing_quality int, label string, split string")
        for split in SPLITS:
            filenames = list(Path(f"/{DATA_VOLUME}/{split}").glob("*.inkml"))
            random.shuffle(filenames)
            filenames = filenames[:N_SAMPLES]
            # split_stats = analyze_ink.map(filenames)
            split_stats = [analyze_ink.local(filename) for filename in filenames]
            split_df = spark.createDataFrame(
                [
                    Row(
                        img_path=str(stat["img_path"]),
                        writing_quality=stat["writing_quality"],
                        label=stat["label"],
                        split=split,
                    )
                    for stat in split_stats
                ],
                schema=df.schema,
            )
            new_entries = split_df.join(df, on=["img_path", "split"], how="left_anti")
            df = df.unionByName(new_entries)
            print(f"Collected {split_df.count()} samples for split {split}")

        # save df for later use
        df.write.mode("overwrite").parquet(f"/{DATA_VOLUME}/data.parquet")

        # save img paths to folders: f"/{DATA_VOLUME}/cls/{split}/{writing_quality}/
        for split in SPLITS:
            for quality in QUALITIES:
                quality_df = df.filter(col("split") == split).filter(col("writing_quality") == quality)
                quality_path = Path(f"/{DATA_VOLUME}/cls/{split}/{quality}")
                if not quality_path.exists():
                    quality_path.mkdir(parents=True)
                for row in quality_df.collect():
                    shutil.copy(row.img_path, quality_path)

    if sft:
        # load df
        df = spark.read.parquet(f"/{DATA_VOLUME}/data.parquet")

        # run trained classifier on df

        # filter to only get data with writing quality == 1
        filtered_dfs = []
        for split in SPLITS:
            split_df = df.filter(col("split") == split)
            split_filter_df = split_df.filter(split_df.writing_quality == 1)
            filtered_dfs.append(split_filter_df)
            print(f"Found {split_filter_df.count()} samples with writing quality == 1 for split {split}")
        filter_df = filtered_dfs[0]
        for filtered_df in filtered_dfs[1:]:
            filter_df = filter_df.unionByName(filtered_df)
        df = filter_df

        # deduplication
        dedup_dfs = []
        for split in SPLITS:
            split_df = df.filter(col("split") == split)
            split_dedup_df = dedup_per_split(split_df)
            dedup_dfs.append(split_dedup_df)
            print(f"Found {split_dedup_df.count()} deduplicated samples for split {split}")
        final_dedup_df = dedup_dfs[0]
        for dedup_df in dedup_dfs[1:]:
            final_dedup_df = final_dedup_df.unionByName(dedup_df)
        df = final_dedup_df

        # write to json
        for split in SPLITS:
            output_path = Path(f"/{DATA_VOLUME}/{split}/data.json")
            split_df = df.filter(col("split") == split)
            json_output = []
            for row in split_df.collect():
                json_entry = {
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
                json_output.append(json_entry)
            with open(output_path, "w") as f:
                json.dump(json_output, f, indent=4)
            print(f"Deduplicated data written to {output_path}")

    if dpo:
        # load df

        # run trained VLM on val data

        # identify subset of val samples with worst perf

        # prompt user to manually label data

        # write to json
        pass


@app.local_entrypoint()
def main(cls: bool = False, sft: bool = False, dpo: bool = False):
    if not cls and not sft and not dpo:
        raise ValueError("Must specify at least one of `cls`, `sft`, or `dpo`")
    run.local(cls, sft, dpo)
