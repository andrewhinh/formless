import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# import re
from xml.etree import ElementTree

import modal

from utils import DATA_VOLUME, GPU_IMAGE, MINUTES, NAME, SECRETS, VOLUME_CONFIG

# -----------------------------------------------------------------------------

MODEL = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
QUANTIZATION = "awq"
ENFORCE_EAGER = True
MAX_NUM_SEQS = 1

WRITE_TEMPERATURE = 0.2
WRITE_MAX_TOKENS = 5
WRITING_QUALITY_PROMPT = """
You are given an image of a math expression and its corresponding annotation.

Determine the quality of the handwriting in the image using the additive 3-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
1) Add 1 point if the writing is completely unreadable.
2) Add another point if the writing is partially unreadable.
3) Add a third point if the writing is completely clear and legible.

Return the quality of the handwriting as a number between 1 and 3.
"""

QUERY_TEMPERATURE = 1.0
QUERY_MAX_TOKENS = 5
USER_QUERY_PROMPT = """
You are a user of a handwriting OCR app, and have an image you want annotated.
Come up with a question that you might ask to get the OCR annotation for this image.
To make this more interesting, flip a coin.
If heads, simply ask what the image contains.
If tails, be creative with the question above: include typos, vagueness, etc.
"""

SAMPLES_PER_GROUP = 10
MINHASH_THRESHOLD = 0.8
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
    GPU_IMAGE.apt_install(["libcairo2-dev", "libjpeg-dev", "libgif-dev"])
    .pip_install(
        "vllm==0.6.5",
        "ninja==1.11.1",  # required to build flash-attn
        "packaging==23.1",  # required to build flash-attn
        "wheel==0.41.2",  # required to build flash-attn
        "torch==2.5.1",  # required to build flash-attn
        "pycairo==1.27.0",
        # "jiwer==3.0.5",
        "pydantic==2.10.4",
        "tqdm==4.67.1",
        "datasketch==1.6.5",
        "ImageHash==4.3.1",
    )
    .run_commands(  # add flash-attn
        "pip install flash-attn==2.7.2.post1 --no-build-isolation"
    )
    .run_function(
        download_model,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
)
ETL_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "t4"
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

    # import jiwer
    import numpy as np
    import PIL
    import PIL.Image
    from datasketch import MinHash, MinHashLSH
    from imagehash import phash
    from PIL import Image
    from pydantic import Field
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def analyze_ink(filename: Path) -> dict:  # noqa: C901
    # setup
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

    def read_inkml_file(filename: str) -> Ink:
        """Simple reader for MathWriting's InkML files."""
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

        return Ink(strokes=strokes, annotations=annotations)

    def cairo_to_pil(surface: cairo.ImageSurface) -> PIL.Image.Image:
        """Converts a ARGB Cairo surface into an RGB PIL image."""
        size = (surface.get_width(), surface.get_height())
        stride = surface.get_stride()
        with surface.get_data() as memory:
            return PIL.Image.frombuffer("RGB", size, memory.tobytes(), "raw", "BGRX", stride)

    def render_ink(
        ink: Ink,
        *,
        margin: int = 10,
        stroke_width: float = 1.5,
        stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
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

        return cairo_to_pil(surface)

    llm = LLM(
        model=MODEL,
        enforce_eager=ENFORCE_EAGER,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=GPU_COUNT,
        **({"quantization": QUANTIZATION} if QUANTIZATION is not None else {}),
    )

    stop_token_ids = None
    write_samp_params = SamplingParams(
        temperature=WRITE_TEMPERATURE,
        max_tokens=WRITE_MAX_TOKENS,
        stop_token_ids=stop_token_ids,
        guided_decoding=GuidedDecodingParams(choice=[1, 2, 3]),
    )
    query_samp_params = SamplingParams(
        temperature=QUERY_TEMPERATURE,
        max_tokens=QUERY_MAX_TOKENS,
        stop_token_ids=stop_token_ids,
    )

    def model_call(prompt: str, samp_params: SamplingParams, img_url: str = None):
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
        outputs = llm.chat(messages, samp_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text

    # run
    ink = read_inkml_file(filename)
    img_path = filename.with_suffix(".png")
    img = render_ink(ink)
    if os.path.exists(img_path):
        os.remove(img_path)
    img.save(img_path)
    with open(img_path, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{base64_img}"
    writing_quality = int(model_call(WRITING_QUALITY_PROMPT, write_samp_params, img_url))
    user_query = model_call(USER_QUERY_PROMPT, query_samp_params, img_url)
    label = ink.annotations["label"]
    return {"img_url": img_url, "writing_quality": writing_quality, "user_query": user_query, "label": label}


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def run():
    # collect metadata
    metadata = {}
    for split in ["train", "valid", "test"]:
        filenames = []
        file_list = Path(f"/{DATA_VOLUME}/{split}").glob("*.inkml")
        for filename in tqdm(file_list, desc=f"Processing {split}", unit="file"):
            filenames.append(filename)
        split_stats = list(analyze_ink.map(filenames))
        metadata[split] = split_stats

    # filter to only get data with writing quality == 1
    filtered = {}
    for split, stats in metadata.items():
        filtered[split] = []
        for stat in stats:
            if stat["writing_quality"] == 1:
                filtered[split].append(stat)

    # stratified deduplication
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERM)  # LSH for near-duplicates
    dedup = {}
    for split, stats in filtered.items():
        dedup[split] = []
        for stat in stats:
            img_hash = phash(Image.open(stat["img_url"]), hash_size=HASH_SZ)
            m = MinHash(num_perm=NUM_PERM)
            m.update(str(img_hash).encode("utf8"))

            duplicates = lsh.query(m)
            if not duplicates:
                lsh.insert(f"{split}_{len(dedup[split])}", m)
                dedup[split].append(stat)
            else:
                existing_stat = dedup[split][duplicates[0]]
                if stat["writing_quality"] > existing_stat["writing_quality"]:
                    dedup[split][duplicates[0]] = stat
        dedup[split] = sorted(dedup[split], key=lambda x: x["writing_quality"], reverse=True)[:SAMPLES_PER_GROUP]

    # write to jsonl
    id = 0
    for split in dedup.keys():
        with open(Path(f"/{DATA_VOLUME}/{split}/data.jsonl"), "w") as f:
            for item in dedup[split]:
                item["id"] = id
                item["conversations"] = [
                    {
                        "from": "user",
                        "value": f"Picture 1: <img>{item['img_url']}</img>\n{item['user_query']}",
                    },
                    {
                        "from": "assistant",
                        "value": item["label"],
                    },
                ]
                f.write(json.dumps(item) + "\n")
                id += 1

    # TODO: use to create tokens and test model
    # _COMMAND_RE = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")

    # def tokenize_expression(s: str) -> list[str]:
    #     r"""Transform a Latex math string into a list of tokens.

    #     Tokens are strings that are meaningful in the context of Latex
    #     e.g. '1', r'\alpha', r'\frac'.

    #     Args:
    #       s: unicode input string (ex: r"\frac{1}{2}")

    #     Returns
    #     -------
    #       tokens: list of tokens as unicode strings.
    #     """
    #     tokens = []
    #     while s:
    #         if s[0] == "\\":
    #             tokens.append(_COMMAND_RE.match(s).group(0))
    #         else:
    #             tokens.append(s[0])

    #         s = s[len(tokens[-1]) :]

    #     return tokens

    # # Example
    # print(tokenize_expression(r"\frac{\alpha}{2}\begin{matrix}1&0\\0&1\end{matrix}\not\in\mathbb{R}"))
    # ### CER Computation

    # def compute_cer(truth_and_output: list[tuple[str, str]]):
    #     """Computes CER given pairs of ground truth and model output."""

    #     class TokenizeTransform(jiwer.transforms.AbstractTransform):
    #         def process_string(self, s: str):
    #             return tokenize_expression(r"{}".format(s))

    #         def process_list(self, tokens: list[str]):
    #             return [self.process_string(token) for token in tokens]

    #     ground_truth, model_output = zip(*truth_and_output, strict=False)

    #     return jiwer.cer(
    #         truth=list(ground_truth),
    #         hypothesis=list(model_output),
    #         reference_transform=TokenizeTransform(),
    #         hypothesis_transform=TokenizeTransform(),
    #     )

    # # Test data to run compute_cer().
    # # The first element is the model prediction, the second the ground truth.
    # examples = [
    #     (r"\sqrt{2}", r"\sqrt{2}"),  # 0 mistakes, 4 tokens
    #     (r"\frac{1}{2}", r"\frac{i}{2}"),  # 1 mistake, 7 tokens
    #     (r"\alpha^{2}", "a^{2}"),  # 1 mistake, 5 tokens
    #     ("abc", "def"),  # 3 mistakes, 3 tokens
    # ]

    # # 5 mistakes for 19 tokens: 26.3% error rate.
    # print(f"{compute_cer(examples)*100:.1f} %")


@app.local_entrypoint()
def main():
    run.remote()
