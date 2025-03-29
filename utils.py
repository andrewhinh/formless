import base64
import io
import os
import random
import subprocess
import tempfile
import warnings
from pathlib import Path, PurePosixPath

import modal
import requests
import validators
from dotenv import load_dotenv
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

APP_NAME = "formless"

PARENT_PATH = Path(__file__).parent
EG_PATH = PARENT_PATH / "eg"
ARTIFACTS_PATH = PARENT_PATH / "training" / "artifacts"
SRC_PATH = PARENT_PATH / "src"


# Terminal image
warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


# image validation
def validate_image_url(image_url: str) -> dict[str, str]:
    if image_url:
        if not validators.url(image_url):
            return {"error": "Invalid image URL"}
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to fetch image: {str(e)}"}

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return {"error": "URL does not point to an image"}

        image_data = response.content
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        return validate_image_base64(image_base64)
    return {"error": "No image URL provided"}


def validate_image_file(
    image_file,
) -> dict[str, str]:
    if image_file is not None:
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        file_extension = Path(image_file.filename).suffix.lower()
        if file_extension not in valid_extensions:
            return {"error": "Invalid file type. Please upload an image."}
        image_file.file.seek(0)  # reset pointer in case of multiple uploads
        img_bytes = image_file.file.read()
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return validate_image_base64(image_base64)
    return {"error": "No image uploaded"}


MAX_FILE_SIZE_MB = 20
MAX_DIMENSIONS = (4096, 4096)


def validate_image_base64(image_base64: str) -> dict[str, str]:
    # Verify MIME type and magic #
    img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    try:
        img.verify()
    except Exception as e:
        return {"error": e}

    # Limit img size
    if len(image_base64) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return {"error": f"File size exceeds {MAX_FILE_SIZE_MB}MB limit."}
    if img.size[0] > MAX_DIMENSIONS[0] or img.size[1] > MAX_DIMENSIONS[1]:
        return {"error": f"Image dimensions exceed {MAX_DIMENSIONS[0]}x{MAX_DIMENSIONS[1]} pixels limit."}

    # Run antivirus
    # write image_base64 to tmp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(base64.b64decode(image_base64))
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(  # noqa: S603
            ["python", "main.py", str(tmp_file_path)],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PARENT_PATH / "Python-Antivirus",
        )
        scan_result = result.stdout.strip().lower()
        if scan_result == "infected":
            return {"error": "Potential threat detected."}
    except Exception as e:
        return {"error": f"Error during antivirus scan: {e}"}

    return {"success": image_base64}


# model
HF_USERNAME = "ajhinh"
CLS_MODEL = f"{APP_NAME}-resnet152-difficulty"
CLS_HF_MODEL = f"{HF_USERNAME}/{CLS_MODEL}"
CLS_HF_RATER = f"hf_hub:{CLS_HF_MODEL}"

PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"

BASE_HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # pretrained model or ckpt
BASE_QUANT_MODEL = f"{BASE_HF_MODEL}-AWQ"
SFT_MODEL = "qwen2.5-vl-7b-instruct-full-sft"
SFT_HF_MODEL = f"{HF_USERNAME}/{APP_NAME}-{SFT_MODEL}"  # pretrained model or ckpt
SFT_QUANT_MODEL = f"{SFT_HF_MODEL}-awq"

DPO_MODEL = "qwen2.5-vl-7b-instruct-lora-dpo"
DPO_MERGED = f"{DPO_MODEL}-merged"
DPO_HF_MODEL = f"{HF_USERNAME}/{APP_NAME}-{DPO_MERGED}"
DPO_QUANT_MODEL = f"{DPO_HF_MODEL}-awq"


# vlm config
QUANTIZATION = "awq_marlin"  # "awq_marlin"
KV_CACHE_DTYPE = None  # "fp8_e5m2"
LIMIT_MM_PER_PROMPT = {"image": 1}
ENFORCE_EAGER = False
MAX_NUM_SEQS = 1
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TEMPERATURE = 0.0
TOP_P = 0.001
REPEATION_PENALTY = 1.05
STOP_TOKEN_IDS = []
MAX_MODEL_LEN = 32768

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


def run_model(
    img_paths: list[Path],
    model: str,
    prompt: str,
    quant: bool,
    gpu_count: int,
    max_tokens: int = None,
    guided_decoding: GuidedDecodingParams = None,
) -> list[str]:
    """
    Run model on image(s) with prompt.
    """
    # load pretrained vlm if not already loaded
    global quantization, llm, sampling_params
    if "quantization" not in globals():
        quantization = "awq_marlin" if quant else None
    if "llm" not in globals():
        llm = LLM(
            model=model,
            tokenizer=PROCESSOR,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            enforce_eager=ENFORCE_EAGER,
            max_num_seqs=MAX_NUM_SEQS,
            tensor_parallel_size=gpu_count,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            mm_processor_kwargs={
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            },
            **{
                k: v
                for k, v in [
                    ("quantization", quantization),
                    ("kv_cache_dtype", KV_CACHE_DTYPE),
                ]
                if v is not None
            },
        )
    if "sampling_params" not in globals():
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPEATION_PENALTY,
            stop_token_ids=STOP_TOKEN_IDS,
            max_tokens=max_tokens,
            guided_decoding=guided_decoding,
        )

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

    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    preds = [out.outputs[0].text.strip() for out in outputs]
    return preds


# data
SPLITS = ["train", "valid", "test"]
CLASSES = ["1", "2", "3"]

# inference
DEFAULT_IMG_URL = "https://formless-data.s3.us-west-1.amazonaws.com/0.png"
DEFAULT_IMG_PATHS = list(EG_PATH.glob("*.png"))
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "What is the content of this image?"

# Modal
IN_PROD = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"
load_dotenv(".env" if IN_PROD else ".env.dev")
SECRETS = [modal.Secret.from_dotenv(path=PARENT_PATH, filename=".env" if IN_PROD else ".env.dev")]

CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"
PYTHON_VERSION = "3.12"

PRETRAINED_VOLUME = f"{APP_NAME}-pretrained"
DATA_VOLUME = f"{APP_NAME}-data"
RUNS_VOLUME = f"{APP_NAME}-runs"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(PRETRAINED_VOLUME, create_if_missing=True),
    f"/{DATA_VOLUME}": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    f"/{RUNS_VOLUME}": modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
}

if modal.is_local():
    DB_VOL_PATH = PARENT_PATH / "local_db"
    DATA_VOL_PATH = ARTIFACTS_PATH / "mathwriting-2024-excerpt"
    RUNS_VOL_PATH = ARTIFACTS_PATH / "runs"
    DB_VOL_PATH.mkdir(parents=True, exist_ok=True)
    DATA_VOL_PATH.mkdir(parents=True, exist_ok=True)
    RUNS_VOL_PATH.mkdir(parents=True, exist_ok=True)
else:
    DATA_VOL_PATH = Path(f"/{DATA_VOLUME}") / "mathwriting-2024-excerpt"
    RUNS_VOL_PATH = Path(f"/{RUNS_VOLUME}")

CPU = 8  # cores
MEM = 32768  # MB
MINUTES = 60  # seconds

TRAIN_REPO_PATH = Path("/LLaMA-Factory")
GPU_IMAGE = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        TAG, add_python=PYTHON_VERSION
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add Python dependencies
        "accelerate>=0.34.0,<=1.2.1",
        "bitsandbytes>=0.45.1",
        "deepspeed>=0.16.3",
        "fastapi>=0.115.6",
        "hf-transfer>=0.1.9",
        "huggingface-hub>=0.28.1",
        "jiwer>=3.1.0",
        "more-itertools>=10.6.0",
        "ninja>=1.11.1.3",  # required to build flash-attn
        "packaging>=24.2",  # required to build flash-attn
        "psycopg2-binary>=2.9.10",
        "pyyaml>=6.0.2",
        "requests>=2.32.3",
        "sqlmodel>=0.0.22",
        "term-image>=0.7.2",
        "timm>=1.0.14",
        "torch>=2.5.1",
        "tqdm>=4.67.1",
        "transformers @ git+https://github.com/huggingface/transformers.git@9985d06add07a4cc691dc54a7e34f54205c04d40",
        "validators>=0.34.0",
        "vllm>=0.7.2",
        "wandb>=0.19.6",
        "wheel>=0.45.1",  # required to build flash-attn
    )
    .run_commands("pip install git+https://github.com/seungwoos/AutoAWQ.git@add-qwen2_5_vl --no-deps")
    .run_commands(  # add flash-attn
        "pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
    .run_commands(  # antivirus for file uploads
        ["git clone https://github.com/Len-Stevens/Python-Antivirus.git /root/Python-Antivirus"]
    )
    .run_commands(
        [
            f"git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git {TRAIN_REPO_PATH}",
            f"cd {TRAIN_REPO_PATH} && pip install -e '.[torch,metrics]'",
        ]
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HUGGINGFACE_HUB_CACHE": f"/{PRETRAINED_VOLUME}",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "FORCE_TORCHRUN": "1",
            "WANDB_PROJECT": APP_NAME,
        }
    )
    .add_local_dir(PARENT_PATH / "db", "/root/db")
)


## subprocess for Modal
def _exec_subprocess(cmd: list[str]):
    """Executes subprocess and prints log to terminal while subprocess is running."""
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            line_str = line.decode()
            print(f"{line_str}", end="")

    if exitcode := process.wait() != 0:
        raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))
