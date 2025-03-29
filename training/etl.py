"""ETL for data to train classifiers and VLMs."""

import hashlib
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
import pandas as pd
import torch
import yaml
from datasketch import MinHash, MinHashLSH
from imagehash import phash
from more_itertools import chunked
from PIL import Image, ImageFile, ImageFilter
from pydantic import Field
from timm.data import create_transform, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import set_jit_fuser, setup_default_logging
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from vllm.sampling_params import GuidedDecodingParams

from utils import (
    APP_NAME,
    BASE_QUANT_MODEL,
    CLASSES,
    CLS_HF_RATER,
    CPU,
    DATA_VOL_PATH,
    DEFAULT_USER_PROMPT,
    GPU_IMAGE,
    MAX_NUM_SEQS,
    MAX_PIXELS,
    MEM,
    MIN_PIXELS,
    MINUTES,
    RANDOM_SEED,
    SECRETS,
    SFT_QUANT_MODEL,
    SPLITS,
    VOLUME_CONFIG,
    run_model,
)

# setup
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger("timm").setLevel(
    logging.WARNING
)  # disable "Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.""

# -----------------------------------------------------------------------------

# classifier training data config

MAX_SCORE_TOKENS = 3

N_SAMPLES_PER_SPLIT_CLS = {
    "train": 5,
    "valid": 5,
    "test": 5,
}

DIFFICULTY_PROMPT = """
You are given an image of a handwritten math expression and its corresponding annotation.
Determine the grade level of the expression in the image using the additive 3-point scoring system described below.
Points are accumulated based on the satisfaction of each criterion:
1) Add 1 point if the expression is something that an elementary student could understand.
2) Add another point if the expression is something that only a middle/high school student could understand.
3) Add a third point if the expression is something that only an undergrad/grad/phd student could understand.
Return the difficulty of the expression as a number between 1 and 3 where
1 is the easiest and 3 is the most difficult.
"""

# -----------------------------------------------------------------------------

# sft training data config

CLS_RUN_BS = 1024

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
THRESHOLD = 0.9  # larger = less duplicates
N_SAMPLES_PER_SPLIT_SFT = {
    "train": 8000,
    "valid": 1000,
    "test": 1000,
}  # only train data will be written to json, valid/test will be used for eval

## stitch
STITCH_BS_MAX = 1024  # 1024 ~= 16384 tokens
ROTATE_MAX = 180  # degrees

# -----------------------------------------------------------------------------


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
IMAGE = GPU_IMAGE.apt_install(
    [
        "libcairo2-dev",  # required to build pycairo
        "libjpeg-dev",  # required to build pycairo
        "libgif-dev",  # required to build pycairo
    ]
).pip_install(
    "pycairo>=1.27.0",
    "pydantic>=2.10.4",
    "datasketch>=1.6.5",
    "imagehash>=4.3.2",
    "pandas>=2.2.3",
)
TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "L40S"
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
    cpu=CPU,
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


@app.function(
    image=IMAGE,
    cpu=CPU,
    memory=MEM,
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
        img_paths,
        BASE_QUANT_MODEL,
        DIFFICULTY_PROMPT,
        quant=True,
        gpu_count=GPU_COUNT,
        max_tokens=MAX_SCORE_TOKENS,
        guided_decoding=GuidedDecodingParams(choice=CLASSES),
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
    cpu=CPU,
    memory=MEM,
    gpu=GPU_CONFIG,
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
        transforms, amp_autocast, classifier = setup_classifier(CLS_HF_RATER)
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
    cpu=CPU,
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
                            "value": f"<image>{DEFAULT_USER_PROMPT}",
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
    cpu=CPU,
    memory=MEM,
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
        SFT_QUANT_MODEL,
        DEFAULT_USER_PROMPT,
        quant=True,
        gpu_count=GPU_COUNT,
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
                            "value": f"<image>{DEFAULT_USER_PROMPT}",
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

    CSV_FILENAME = str(DATA_VOL_PATH / "data.csv")

    df = pd.DataFrame([], columns=["img_path", "label", "split"])
    split_cts = {}
    if os.path.exists(CSV_FILENAME):
        print(f"Loading existing dataframe from {CSV_FILENAME}")
        df = pd.read_csv(CSV_FILENAME)
        for split in SPLITS:
            split_cts[split] = len(df[df["split"] == split])
            print(f"Loaded {split_cts[split]} samples for split {split}")
    else:
        print(f"Creating new dataframe from {DATA_VOL_PATH}")
        for split in SPLITS:
            # extract ink metadata
            ink_paths = [Path(p) for p in (DATA_VOL_PATH / split).glob("*.inkml")]
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
            split_df = pd.DataFrame(
                [
                    {
                        "img_path": str(stat["img_path"]),
                        "label": stat["label"],
                        "split": split,
                    }
                    for stat in split_stats
                ]
            )
            split_cts[split] = len(split_df)
            new_entries = split_df[
                ~split_df.set_index(["img_path", "split"]).index.isin(df.set_index(["img_path", "split"]).index)
            ]
            df = pd.concat([df, new_entries], ignore_index=True)
            print(f"Collected {split_cts[split]} samples for split {split}")

        df.to_csv(CSV_FILENAME, index=False)

    if cls:
        # run VLM to assign score to random subset
        df["score"] = None
        df_parts = []
        for split in SPLITS:
            ## get random subset of split data
            split_df = df[df["split"] == split]
            split_df = split_df.sample(
                n=min(N_SAMPLES_PER_SPLIT_CLS[split], len(split_df)), random_state=int(RANDOM_SEED)
            )

            ## run
            img_paths = [Path(row) for row in split_df["img_path"].tolist()]
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
                lst_scores = analyze_inks.map(img_batches)
                scores = [score for batch_scores in lst_scores for score in batch_scores]

            ## save to write later
            split_df = pd.DataFrame({"img_path": [str(p) for p in img_paths], "split": split, "score": scores})
            df_parts.append(split_df)
            print(f"Labeled {len(split_df)} samples for split {split}")

        ## write to df
        if df_parts:
            split_df = pd.concat(df_parts, ignore_index=True)
            df = df.set_index(["img_path", "split"])
            split_df = split_df.set_index(["img_path", "split"])
            df["score"] = split_df["score"].combine_first(df["score"])
            df = df.reset_index()
            df.to_csv(CSV_FILENAME, index=False)
    if sft:
        # run trained classifier on df
        df_parts = []
        for split in SPLITS:
            ## get non-labeled samples
            split_df = df[df["split"] == split]
            split_df = split_df[split_df["score"].isna()]

            ## run model
            img_paths = [Path(row) for row in split_df["img_path"].tolist()]
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
            split_df = pd.DataFrame({"img_path": [str(p) for p in img_paths], "split": split, "score": scores})
            df_parts.append(split_df)

        ## write to df
        if df_parts:
            split_df = pd.concat(df_parts, ignore_index=True)
            df = df.set_index(["img_path", "split"])
            split_df = split_df.set_index(["img_path", "split"])
            df["score"] = split_df["score"].combine_first(df["score"])
            df = df.reset_index()
            df.to_csv(CSV_FILENAME, index=False)

        # deduplication
        df_parts = []

        ## compute minhash vec
        df["phash"] = df["img_path"].apply(lambda img_path: str(phash(Image.open(img_path), hash_size=HASH_SZ)))
        df["minhash"] = df["phash"].apply(compute_minhash)

        ## dedup with lsh
        for split in SPLITS:
            split_df = df[df["split"] == split].copy()

            # Create LSH index for MinHash values
            lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
            for idx, row in split_df.iterrows():
                minhash = MinHash(num_perm=NUM_PERM)
                for val in row["minhash"]:
                    minhash.update(str(val).encode("utf-8"))
                lsh.insert(idx, minhash)

            # Find duplicates
            duplicates = set()
            for idx, row in split_df.iterrows():
                minhash = MinHash(num_perm=NUM_PERM)
                for val in row["minhash"]:
                    minhash.update(str(val).encode("utf-8"))
                similar_items = lsh.query(minhash)
                for similar_idx in similar_items:
                    if similar_idx != idx:
                        duplicates.add(idx)

            # Remove duplicates
            split_df = split_df.drop(list(duplicates))
            split_df = split_df.drop(columns=["phash", "minhash"])
            df_parts.append(split_df)

        # write to df
        if df_parts:
            df = pd.concat(df_parts, ignore_index=True)
            df.to_csv(CSV_FILENAME, index=False)

        # randomly stitch samples together
        img_paths, labels = (
            {split: [] for split in SPLITS},
            {split: [] for split in SPLITS},
        )
        for split in SPLITS:
            split_df = df[df["split"] == split]
            n_samples = N_SAMPLES_PER_SPLIT_SFT[split]
            for score in CLASSES:
                split_filter_df = split_df[split_df["score"] == int(score)]
                percent = len(split_filter_df) / len(split_df)

                paths = [Path(row) for row in split_filter_df["img_path"].tolist()]
                lbls = split_filter_df["label"].tolist()
                if not paths or not lbls:
                    continue

                ## rand sample
                path_batches = []
                combined_labels = []
                for _ in range(int(n_samples * percent)):
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

            print(f"Generated {n_samples} samples for {split} split")

        for split in SPLITS:
            write_sft_json(DATA_VOL_PATH / f"sft_{split}.json", img_paths[split], labels[split])
    if dpo:
        # run model to determine which train samples it fails on
        json_path = DATA_VOL_PATH / "sft_train.json"
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
                    desc="dpo train",
                    total=len(img_batches),
                )
            )
        else:
            lst_preds = list(ft_pred_ink.map(img_batches))
            preds = [item for lst in lst_preds for item in lst]

        img_paths, labels, preds = zip(
            *[
                (path, label, pred)
                for path, label, pred in zip(img_paths, labels, preds, strict=False)
                if label != pred
            ],
            strict=False,
        )
        print(f"Generated {len(img_paths)} DPO samples for train split")
        write_dpo_json(DATA_VOL_PATH / "dpo_train.json", img_paths, preds, labels)


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
    main(args.cls, args.sft, args.dpo)
