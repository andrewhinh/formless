import os
from pathlib import Path

import modal

from utils import (
    DATA_VOLUME,
    DEFAULT_IMG_PATH,
    DEFAULT_IMG_URL,
    DEFAULT_QUESTION,
    GPU_IMAGE,
    IN_PROD,
    MINUTES,
    NAME,
    REMOTE_DB_URI,
    VOLUME_CONFIG,
    Colors,
)

parent_path: Path = Path(__file__).parent
db_dir_path = parent_path.parent / "db"

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
SECRETS = [modal.Secret.from_dotenv(path=parent_path, filename=".env" if IN_PROD else ".env.dev")]
IMAGE = (
    GPU_IMAGE.pip_install(  # add Python dependencies
        "vllm==0.6.2",
        "term-image==0.7.2",
        "fastapi==0.115.6",
        "validators==0.34.0",
        "sqlmodel==0.0.22",
    )
    .run_function(
        download_model,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
    .copy_local_dir(parent_path.parent / "db", "/root/db")
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
    import io
    import os
    import secrets
    import tempfile
    import time
    from contextlib import contextmanager
    from uuid import uuid4

    import requests
    import validators
    from fastapi import FastAPI, HTTPException, Request, Security
    from fastapi.security import APIKeyHeader
    from PIL import Image, ImageFile
    from sqlmodel import Session as DBSession
    from sqlmodel import create_engine, select
    from term_image.image import from_file
    from vllm import LLM, SamplingParams

    from db.models import ApiKey, ApiKeyCreate

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    f_app = FastAPI()
    engine = create_engine(
        url=REMOTE_DB_URI,
        echo=not IN_PROD,
    )

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

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
        engine.dispose()
        VOLUME_CONFIG[f"/{DATA_VOLUME}"].reload()
        with get_db_session() as db_session:
            if db_session.exec(select(ApiKey).where(ApiKey.key == api_key_header)).first() is not None:
                return True
        print(f"Invalid API key: {api_key_header}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    @f_app.post("/")
    async def main(
        image_url: str = DEFAULT_IMG_URL, question: str = DEFAULT_QUESTION, api_key: bool = Security(verify_api_key)
    ) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        if not validators.url(image_url):
            print(f"Invalid request {request_id}: img_url={image_url}, question={question}")
            raise HTTPException(status_code=400, detail="Invalid image URL")

        response = requests.get(image_url, stream=True)

        try:
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        except Exception as e:
            print(f"Error processing request {request_id}: error={str(e)}, img_url={image_url}, question={question}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") from e

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
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"Error processing request {request_id}: error={str(e)}, img_url={image_url}, question={question}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") from e
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

    @f_app.post("/upload")
    async def main_upload(
        request: Request, question: str = DEFAULT_QUESTION, api_key: bool = Security(verify_api_key)
    ) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_data = await request.body()

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error processing request {request_id}: error={str(e)}, question={question}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}") from e

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
        image_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.jpg")
        image.save(image_path)
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
        k = ApiKeyCreate(key=secrets.token_hex(16))
        k = ApiKey.model_validate(k)
        with get_db_session() as db_session:
            db_session.add(k)
            db_session.commit()
            db_session.refresh(k)
        return k.key

    return f_app


## For testing
@app.local_entrypoint()
def main():
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

    response = requests.post(
        f"{modal_get.web_url}/upload",
        data=open(DEFAULT_IMG_PATH, "rb").read(),
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/octet-stream",
            "question": DEFAULT_QUESTION,
        },
    )
    assert response.ok, response.status_code


# TODO:
# - add multiple uploads/urls

# - Replace with custom model impl FT on hard images
# - Add custom CUDA kernels for faster inference
