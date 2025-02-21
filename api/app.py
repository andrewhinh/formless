import base64
import os
import secrets
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import modal
import requests
import torch
import uvicorn
import validators
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image, ImageFile
from pydantic import BaseModel
from sqlmodel import Session as DBSession
from sqlmodel import create_engine, select
from term_image.image import from_file
from vllm import LLM, SamplingParams

from db.models import ApiKey, ApiKeyCreate
from utils import (
    APP_NAME,
    DB_VOL_PATH,
    DEFAULT_IMG_PATH,
    DEFAULT_IMG_URL,
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    GPU_IMAGE,
    HF_USERNAME,
    IN_PROD,
    MINUTES,
    PARENT_PATH,
    SECRETS,
    VOLUME_CONFIG,
    Colors,
)

# -----------------------------------------------------------------------------

MODEL = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-dpo-merged-awq"  # pretrained model or ckpt
TOKENIZER = "Qwen/Qwen2-VL-7B-Instruct"  # pretrained tokenizer
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
MAX_MODEL_LEN = 16384

MAX_FILE_SIZE_MB = 5
MAX_DIMENSIONS = (4096, 4096)


load_dotenv(".env" if IN_PROD else ".env.dev")


def download_model():
    from huggingface_hub import login, snapshot_download

    if not os.path.exists(MODEL):
        login(token=os.getenv("HF_TOKEN"), new_session=False)
        snapshot_download(
            MODEL,
            ignore_patterns=["*.pt", "*.bin"],
        )
    else:  # check if preprocessor_config.json was successfully copied; if not, do so
        if not os.path.exists(f"{MODEL}/preprocessor_config.json"):
            login(token=os.getenv("HF_TOKEN"), new_session=False)
            tok_path = snapshot_download(
                TOKENIZER,
                ignore_patterns=["*.pt", "*.bin"],
            )
            os.rename(f"{tok_path}/preprocessor_config.json", f"{MODEL}/preprocessor_config.json")


if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 1


def get_app():  # noqa: C901
    ## setup
    f_app = FastAPI()
    engine = create_engine(
        url=os.getenv("POSTGRES_URL"),
        echo=not IN_PROD,
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    upload_dir = Path(f"{DB_VOL_PATH}/uploads")
    upload_dir.mkdir(exist_ok=True)

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

    llm = LLM(
        model=MODEL,
        tokenizer=TOKENIZER,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        enforce_eager=ENFORCE_EAGER,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=GPU_COUNT,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
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
        stop_token_ids=STOP_TOKEN_IDS,
    )

    ## helpers

    async def verify_api_key(
        api_key_header: str = Security(APIKeyHeader(name="X-API-Key")),
    ) -> bool:
        with get_db_session() as db_session:
            if db_session.exec(select(ApiKey).where(ApiKey.key == api_key_header)).first() is not None:
                return True
        print(f"Invalid API key: {api_key_header}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    class UrlInput(BaseModel):
        image_url: str

    ## main

    @f_app.post("/")
    async def main(input_data: UrlInput, api_key: bool = Security(verify_api_key)) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = input_data.image_url

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
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_QUESTION},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        outputs = llm.chat(messages, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        ## print response
        try:
            ext: str = image_url.split("/")[-1].split(".")[-1]
            upload_path = upload_dir / f"{uuid4()}.{ext}"
            upload_path.write_bytes(response.content)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}") from e

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

    @f_app.post("/upload")
    async def main_upload(image: UploadFile, api_key: bool = Security(verify_api_key)) -> str:
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
        upload_path = upload_dir / f"{uuid4()}.{file_extension}"
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
                cwd=PARENT_PATH / "Python-Antivirus",
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
        with open(upload_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_img}"
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_QUESTION},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        outputs = llm.chat(messages, sampling_params)
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
        k = ApiKeyCreate(key=secrets.token_hex(32), session_id=str(uuid4()))
        k = ApiKey.model_validate(k)
        with get_db_session() as db_session:
            db_session.add(k)
            db_session.commit()
            db_session.refresh(k)
        return k.key

    return f_app


# -----------------------------------------------------------------------------


# Modal
IMAGE = (
    GPU_IMAGE.pip_install(  # add Python dependencies
        "term-image==0.7.2",
        "fastapi==0.115.6",
        "validators==0.34.0",
        "sqlmodel==0.0.22",
        "psycopg2-binary==2.9.10",
    )
    .run_commands(["git clone https://github.com/Len-Stevens/Python-Antivirus.git /root/Python-Antivirus"])
    .run_function(
        download_model,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
    .add_local_dir(PARENT_PATH / "db", "/root/db")
    .add_local_python_source("db", "utils")
)
API_TIMEOUT = 5 * MINUTES
API_CONTAINER_IDLE_TIMEOUT = 15 * MINUTES  # max
API_ALLOW_CONCURRENT_INPUTS = 1000  # max

GPU_TYPE = "l4"
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

app = modal.App(name=f"{APP_NAME}-api")

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
    return get_app()


## For testing
@app.local_entrypoint()
def main():
    import requests

    response = requests.post(f"{modal_get.web_url}/api-key")
    assert response.ok, response.status_code
    api_key = response.json()

    response = requests.post(
        f"{modal_get.web_url}/",
        json={"image_url": DEFAULT_IMG_URL},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code

    response = requests.post(
        f"{modal_get.web_url}/upload",
        files={"image": open(DEFAULT_IMG_PATH, "rb")},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code


if __name__ == "__main__":
    download_model()
    uvicorn.run(get_app(), reload=True)

# TODO
# - add multiple uploads/urls
# - add user authentication:
#   - save gens and keys to user account
#   - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#       - Only allow authorized users to upload files: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
