import os
import subprocess
import warnings
from pathlib import Path, PurePosixPath

import modal
from huggingface_hub import HfApi

APP_NAME = "formless"
DEFAULT_IMG_URL = "https://www.researchgate.net/publication/328401414/figure/fig1/AS:941482479472661@1601478323514/Handwritten-mathematical-expression-example.png"
PARENT_PATH = Path(__file__).parent
DEFAULT_IMG_PATH = PARENT_PATH / "api" / "eg.png"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_QUESTION = "What is the content of this image?"

# HF
HF_USERNAME = HfApi().whoami(token=os.getenv("HF_TOKEN"))["name"]

# Modal
IN_PROD = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"
SECRETS = [modal.Secret.from_dotenv(path=PARENT_PATH, filename=".env" if IN_PROD else ".env.dev")]

CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"
PYTHON_VERSION = "3.12"

PRETRAINED_VOLUME = f"{APP_NAME}-pretrained"
DB_VOLUME = f"{APP_NAME}-db"
DATA_VOLUME = f"{APP_NAME}-data"
RUNS_VOLUME = f"{APP_NAME}-runs"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(PRETRAINED_VOLUME, create_if_missing=True),
    f"/{DB_VOLUME}": modal.Volume.from_name(DB_VOLUME, create_if_missing=True),
    f"/{DATA_VOLUME}": modal.CloudBucketMount(DATA_VOLUME, secret=SECRETS[0]),
    f"/{RUNS_VOLUME}": modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
}


if modal.is_local():
    DB_VOL_PATH = str(PARENT_PATH / "local_db")
    if not os.path.exists(DB_VOL_PATH):
        os.mkdir(DB_VOL_PATH)
        os.chmod(DB_VOL_PATH, 0o777)  # noqa: S103
else:
    DB_VOL_PATH = f"/{DB_VOLUME}"

CPU = 20  # cores (Modal max)
MINUTES = 60  # seconds

GPU_IMAGE = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        TAG, add_python=PYTHON_VERSION
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add Python dependencies
        "vllm==0.7.1",
        "huggingface_hub[hf_transfer]==0.29.1",
        "ninja==1.11.1",  # required to build flash-attn
        "packaging==23.1",  # required to build flash-attn
        "wheel==0.45.1",  # required to build flash-attn
        "torch==2.5.1",  # required to build flash-attn,
    )
    .run_commands(  # add flash-attn
        "pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HUGGINGFACE_HUB_CACHE": f"/{PRETRAINED_VOLUME}",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
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
