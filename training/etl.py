import json
import math
import os
from pathlib import Path

# import re
from xml.etree import ElementTree

import modal

from utils import (
    DATA_VOLUME,
    GPU_IMAGE,
    IN_PROD,
    MINUTES,
    NAME,
    PARENT_PATH,
    VOLUME_CONFIG,
)

# -----------------------------------------------------------------------------

model = "gpt-4o"
seed = 42
temperature = 0.0
max_tokens = 1

prompt = """
You are given an image of a math expression and its corresponding annotation.
Your task is to determine what topic the expression is about and what level of difficulty it is.
For example, if the expression contains "a+b=c^2", you would determine the topic is "Algebra" and the level is "High School".
You will also determine the quality of the handwriting in the image and return a score.

Here is the process for determining the quality of the handwriting:
1. Add 1 point if the writing is clear but not legible.
2. Add 1 point if the writing is legible but not clear.
3. Add 1 point if the writing is clear and legible.
"""


# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, tuple, Path, type(None)))
]
config = {k: globals()[k] for k in config_keys}
config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}  # since Path not serializable

# -----------------------------------------------------------------------------


# Modal
SECRETS = [modal.Secret.from_dotenv(path=PARENT_PATH, filename=".env" if IN_PROD else ".env.dev")]
IMAGE = GPU_IMAGE.pip_install(
    "cairo",
    "matplotlib",
    # "jiwer",
    "numpy",
    "Pillow",
    "openai",
    "pydantic",
    "tqdm",
).copy_local_dir(
    PARENT_PATH / "training" / "artifacts",
    f"/{DATA_VOLUME}/artifacts",
)


ETL_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "H100"
GPU_COUNT = 2
GPU_MEMORY = None  # options = None, 40, 80
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(memory=GPU_MEMORY, count=GPU_COUNT)

APP_NAME = f"{NAME}-etl"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------


@app.function(
    image=GPU_IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=ETL_TIMEOUT,
)
def run():  # noqa: C901
    import cairo
    import matplotlib.pyplot as plt

    # import jiwer
    import numpy as np
    import PIL
    import PIL.Image
    from openai import OpenAI
    from PIL import ImageFile
    from pydantic import BaseModel, Field
    from tqdm import tqdm

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    MATHWRITING_ROOT_DIR = f"/{DATA_VOLUME}/artifacts/mathwriting-2024"

    class Ink(BaseModel):
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

    class Response:
        topic: str
        level: str
        writing_quality: int

    def openai_call(img: PIL.Image.Image):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        messages[1]["content"].append({"type": "image", "image": {"data": img.tobytes()}})

        return (
            client.beta.chat.completions.parse(
                model="gpt-4o",
                temperature=config["temperature"],
                seed=config["seed"],
                messages=messages,
                response_format=Response,
            )
            .choices[0]
            .message.parsed
        )

    stats = {
        "inkCreationMethod": {},
        "labelLength": {},
        "normalizedLabelLength": {},
        "imgWidth": {},
        "imgHeight": {},
        "topic": {},
        "level": {},
        "writingQuality": {},
        "split": {},
    }

    for split in ["train", "valid", "test"]:
        file_list = os.listdir(os.path.join(MATHWRITING_ROOT_DIR, split))
        for filename in tqdm(file_list, desc=f"Processing {split}", unit="file"):
            ink = read_inkml_file(os.path.join(MATHWRITING_ROOT_DIR, split, filename))

            annotations = ink.annotations
            method = annotations.get("inkCreationMethod", "unknown")
            stats["inkCreationMethod"][method] = stats["inkCreationMethod"].get(method, 0) + 1
            label_length = len(annotations.get("label", ""))
            stats["labelLength"][label_length] = stats["labelLength"].get(label_length, 0) + 1
            norm_label_length = len(annotations.get("normalizedLabel", ""))
            stats["normalizedLabelLength"][norm_label_length] = (
                stats["normalizedLabelLength"].get(norm_label_length, 0) + 1
            )

            img = render_ink(ink)
            width, height = img.size
            stats["imgWidth"][width] = stats["imgWidth"].get(width, 0) + 1
            stats["imgHeight"][height] = stats["imgHeight"].get(height, 0) + 1

            response = openai_call(img)
            stats["topic"][response.topic] = stats["topic"].get(response.topic, 0) + 1
            stats["level"][response.level] = stats["level"].get(response.level, 0) + 1
            stats["writingQuality"][response.writing_quality] = (
                stats["writingQuality"].get(response.writing_quality, 0) + 1
            )
            stats["split"] = stats.get("split", {})
            stats["split"][split] = stats["split"].get(split, 0) + 1

    for key, value in stats.items():
        plt.figure(figsize=(10, 5))
        plt.bar(value.keys(), value.values())
        plt.title(f"{key} Distribution")
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        stats_file = os.path.join(MATHWRITING_ROOT_DIR, "stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)

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
