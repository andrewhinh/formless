import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import modal
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from PIL import ImageFile

from utils import (
    DATA_VOLUME,
    DEFAULT_IMG_URL,
    DEFAULT_QUESTION,
    GPU_IMAGE,
    MINUTES,
    NAME,
    VOLUME_CONFIG,
    Colors,
)

parent_path: Path = Path(__file__).parent
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------------

model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
gpu_memory_utilization = 0.90
max_model_len = 8192
max_num_seqs = 1
enforce_eager = True

temperature = 0.2
max_tokens = 1024

# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, tuple, Path, type(None)))
]
config = {k: globals()[k] for k in config_keys}
config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}  # since Path not serializable

# -----------------------------------------------------------------------------


# container build-time fns
def download_model():
    from huggingface_hub import login, snapshot_download

    login(token=os.getenv("HF_TOKEN"), new_session=False)
    snapshot_download(
        config["model"],
        ignore_patterns=["*.pt", "*.bin"],
    )


# Modal
in_prod: bool = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"
SECRETS = [modal.Secret.from_dotenv(path=parent_path, filename=".env" if in_prod else ".env.dev")]
IMAGE = GPU_IMAGE.pip_install(  # add Python dependencies
    "vllm==0.6.2",
    "term-image==0.7.2",
    "python-fasthtml==0.6.10",
    "sqlite-utils==3.18",
    "validators==0.34.0",
).run_function(
    download_model,
    secrets=SECRETS,
    volumes=VOLUME_CONFIG,
)
API_TIMEOUT = 5 * MINUTES
API_CONTAINER_IDLE_TIMEOUT = 1 * MINUTES  # max
API_ALLOW_CONCURRENT_INPUTS = 1000  # max

GPU_TYPE = "H100"
GPU_COUNT = 2
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-api"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------


# Main API
@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=API_TIMEOUT,
    container_idle_timeout=API_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=API_ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():
    import requests
    import validators
    from fasthtml import common as fh
    from PIL import Image
    from term_image.image import from_file
    from vllm import LLM, SamplingParams

    f_app = FastAPI()

    db_path = f"/{DATA_VOLUME}/main.db"
    tables = fh.database(db_path).t
    api_keys = tables.api_keys
    if api_keys not in tables:
        api_keys.create(key=str, granted_at=str, session_id=str, id=int, pk="id")
    ApiKey = api_keys.dataclass()

    llm = LLM(
        model=config["model"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        enforce_eager=config["enforce_eager"],
        tensor_parallel_size=GPU_COUNT,
    )

    async def verify_api_key(
        api_key_header: str = Security(APIKeyHeader(name="X-API-Key")),
    ) -> bool:
        if api_keys(limit=1, where=f"key == '{api_key_header}'") is not None:
            return True
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    @f_app.post("/")
    async def main(request: Request, api_key: bool = Security(verify_api_key)) -> str:
        body = await request.json()

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = body.get("image_url", DEFAULT_IMG_URL)
        if not validators.url(image_url):
            raise HTTPException(status_code=400, detail="Invalid image URL")
        # image_file = body.get("image_file")
        if image_url:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        # else:
        #     image = Image.open(image_file).convert("RGB")

        question = body.get("question", DEFAULT_QUESTION)
        prompt = f"<|image|><|begin_of_text|>{question}"
        stop_token_ids = None

        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stop_token_ids=stop_token_ids,
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # show the question, image, and response in the terminal for demonstration purposes
        response = requests.get(image_url)
        image_filename = image_url.split("/")[-1]
        image_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}-{image_filename}")
        with open(image_path, "wb") as file:
            file.write(response.content)
        terminal_image = from_file(image_path)
        terminal_image.draw()
        print(
            Colors.BOLD,
            Colors.GREEN,
            f"Response: {generated_text}",
            Colors.END,
            sep="",
        )
        print(f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds")

        return generated_text

    @f_app.post("/api-key")
    async def apikey() -> str:
        k = api_keys.insert(ApiKey(key=None, granted_at=None, session_id=None))
        k.key = str(uuid4())
        k.granted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        k.session_id = str(uuid4())
        api_keys.update(k)
        return k.key

    return f_app


## For testing
@app.local_entrypoint()
def main(
    twice=True,
):
    import requests

    response = requests.post(f"{modal_get.web_url}/api-key")
    assert response.ok, response.status_code
    api_key = response.json()

    response = requests.post(
        modal_get.web_url,
        json={"image_url": DEFAULT_IMG_URL, "question": DEFAULT_QUESTION},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code

    if twice:
        # second response is faster, because the Function is already running
        response = requests.post(
            modal_get.web_url,
            json={"image_url": DEFAULT_IMG_URL, "question": DEFAULT_QUESTION},
            headers={"X-API-Key": api_key},
        )
        assert response.ok, response.status_code


# TODO:
# - add file upload security
# - add multiple uploads/urls

# - Replace with custom model impl FT on hard images
# - Add custom CUDA kernels for faster inference
