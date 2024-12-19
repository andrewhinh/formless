import os
from pathlib import Path

import modal

from utils import (
    DB_VOLUME,
    DEFAULT_IMG_PATH,
    DEFAULT_IMG_URL,
    DEFAULT_QUESTION,
    GPU_IMAGE,
    IN_PROD,
    MINUTES,
    NAME,
    PARENT_PATH,
    REMOTE_DB_URI,
    SECRETS,
    VOLUME_CONFIG,
    Colors,
)

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
IMAGE = (
    GPU_IMAGE.pip_install(  # add Python dependencies
        "vllm==0.6.2",
        "term-image==0.7.2",
        "fastapi==0.115.6",
        "validators==0.34.0",
        "sqlmodel==0.0.22",
    )
    .run_commands(["git clone https://github.com/Len-Stevens/Python-Antivirus.git"])
    .run_function(
        download_model,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
    .copy_local_dir(PARENT_PATH / "db", "/root/db")
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
def modal_get():  # noqa: C901
    import os
    import secrets
    import subprocess
    import tempfile
    import time
    from contextlib import contextmanager
    from uuid import uuid4

    import requests
    import validators
    from fastapi import FastAPI, Form, HTTPException, Security, UploadFile
    from fastapi.security import APIKeyHeader
    from PIL import Image, ImageFile
    from pydantic import BaseModel
    from sqlmodel import Session as DBSession
    from sqlmodel import create_engine, select
    from term_image.image import from_file
    from vllm import LLM, SamplingParams

    from db.models import ApiKey, ApiKeyCreate

    ## setup
    f_app = FastAPI()
    engine = create_engine(
        url=REMOTE_DB_URI,
        echo=not IN_PROD,
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    upload_dir = Path(f"/{DB_VOLUME}/uploads")
    upload_dir.mkdir(exist_ok=True)

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

    ## helpers

    async def verify_api_key(
        api_key_header: str = Security(APIKeyHeader(name="X-API-Key")),
    ) -> bool:
        engine.dispose()
        VOLUME_CONFIG[f"/{DB_VOLUME}"].reload()
        with get_db_session() as db_session:
            if db_session.exec(select(ApiKey).where(ApiKey.key == api_key_header)).first() is not None:
                return True
        print(f"Invalid API key: {api_key_header}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    class UrlInput(BaseModel):
        image_url: str
        question: str = DEFAULT_QUESTION

    ## main

    @f_app.post("/")
    async def main(input_data: UrlInput, api_key: bool = Security(verify_api_key)) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = input_data.image_url
        question = input_data.question

        ## validate
        if not image_url or not validators.url(image_url):
            print(f"Invalid image URL: {image_url}")
            raise HTTPException(status_code=400, detail="Invalid image URL")
        response = requests.get(image_url, stream=True)
        try:
            response.raise_for_status()
            pil_image = Image.open(response.raw).convert("RGB")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}") from e

        ## send to model
        prompt = f"<|image|><|begin_of_text|>{question}"
        stop_token_ids = None
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stop_token_ids=stop_token_ids,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": pil_image},
        }

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        ## print response
        response = requests.get(image_url)
        try:
            response.raise_for_status()
            ext: str = image_url.split("/")[-1].split(".")[-1]
            upload_path = upload_dir / f"{uuid4()}{ext}"
            upload_path.write_bytes(response.content)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}") from e

        terminal_image = from_file(upload_path)
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
        image: UploadFile, question: str = Form(...), api_key: bool = Security(verify_api_key)
    ) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_data = image.file.read()

        # validate
        ## ext
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        file_extension = Path(image.filename).suffix.lower()
        if file_extension not in valid_extensions:
            print(f"Invalid file type: {file_extension}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        ## save
        upload_path = upload_dir / f"{uuid4()}{file_extension}"
        upload_path.write_bytes(image_data)
        try:
            pil_image = Image.open(upload_path).convert("RGB")
        except Exception as e:
            print(f"Invalid image data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}") from e

        ## mime type
        try:
            pil_image.verify()
        except Exception as e:
            os.remove(upload_path)
            print(f"Invalid image data: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image data.") from e

        ## size
        MAX_FILE_SIZE_MB = 5
        MAX_DIMENSIONS = (4096, 4096)
        if os.path.getsize(upload_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
            os.remove(upload_path)
            print(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
        if pil_image.size[0] > MAX_DIMENSIONS[0] or pil_image.size[1] > MAX_DIMENSIONS[1]:
            os.remove(upload_path)
            print(f"Image dimensions exceed {MAX_DIMENSIONS[0]}x{MAX_DIMENSIONS[1]} pixels limit.")
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions exceed {MAX_DIMENSIONS[0]}x{MAX_DIMENSIONS[1]} pixels limit.",
            )

        ## antivirus
        try:
            result = subprocess.run(  # noqa: S603
                ["python", "main.py", str(upload_path)],  # noqa: S607
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/Python-Antivirus",
            )
            scan_result = result.stdout.strip().lower()
            if scan_result == "infected":
                os.remove(upload_path)
                print("Potential threat detected.")
                raise HTTPException(status_code=400, detail="Potential threat detected.")
        except Exception as e:
            os.remove(upload_path)
            print(f"Error during antivirus scan: {e}")
            raise HTTPException(status_code=500, detail=f"Error during antivirus scan: {e}") from e

        ## send to model
        prompt = f"<|image|><|begin_of_text|>{question}"
        stop_token_ids = None
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stop_token_ids=stop_token_ids,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": pil_image},
        }

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        ## print response
        image_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.jpg")
        pil_image.save(image_path)
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
        k = ApiKeyCreate(key=secrets.token_hex(16), session_id=str(uuid4()))
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
        f"{modal_get.web_url}/",
        json={"image_url": DEFAULT_IMG_URL, "question": DEFAULT_QUESTION},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code

    response = requests.post(
        f"{modal_get.web_url}/upload",
        files={"image": open(DEFAULT_IMG_PATH, "rb")},
        data={"question": DEFAULT_QUESTION},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code


# TODO
# - Replace with custom model impl FT on hard images
# - Add custom CUDA kernels for faster inference

# - move to postgres
# - add multiple uploads/urls
# - add user authentication:
#   - save gens and keys to user account
#   - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#       - Only allow authorized users to upload files: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
