import base64
import io
import os
import secrets
import tempfile
import time
from contextlib import contextmanager
from uuid import uuid4

import modal
import requests
import torch
import uvicorn
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
    CPU,
    DEFAULT_IMG_PATHS,
    DEFAULT_IMG_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ENFORCE_EAGER,
    GPU_IMAGE,
    KV_CACHE_DTYPE,
    LIMIT_MM_PER_PROMPT,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MAX_PIXELS,
    MEM,
    MIN_PIXELS,
    MINUTES,
    PROCESSOR,
    QUANTIZATION,
    REPEATION_PENALTY,
    SECRETS,
    SFT_QUANT_MODEL,
    STOP_TOKEN_IDS,
    TEMPERATURE,
    TOP_P,
    VOLUME_CONFIG,
    Colors,
    validate_image_file,
    validate_image_url,
)

# -----------------------------------------------------------------------------

# Modal

TIMEOUT = 5 * MINUTES
CONTAINER_IDLE_TIMEOUT = 15 * MINUTES  # max
ALLOW_CONCURRENT_INPUTS = 1000  # max

if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 1

GPU_TYPE = "l4"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

app = modal.App(name=f"{APP_NAME}-api")

# -----------------------------------------------------------------------------

# main


def get_app():  # noqa: C901
    ## setup
    f_app = FastAPI()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    engine = create_engine(
        url=os.getenv("POSTGRES_URL"),
    )

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

    llm = LLM(
        model=SFT_QUANT_MODEL,
        tokenizer=PROCESSOR,
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
        **{
            k: v
            for k, v in [
                ("quantization", QUANTIZATION),
                ("kv_cache_dtype", KV_CACHE_DTYPE),
            ]
            if v is not None
        },
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

    def endpoint(url: str = None, image_file: UploadFile = None) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        ## validate
        if image_file is not None:
            response = validate_image_file(image_file)
        else:
            response = validate_image_url(url)
        if "error" in response.keys():
            msg = response["error"]
            print(msg)
            raise HTTPException(status_code=400, detail=msg)

        ## send to model
        base64_img = list(response.values())[0]
        image_url = f"data:image/jpeg;base64,{base64_img}"
        conversation = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        outputs = llm.chat(conversation, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        ## print response
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image = Image.open(io.BytesIO(base64.b64decode(base64_img)))
            pil_image = pil_image.convert("RGB")
            pil_image.save(tmp_file.name)
            terminal_image = from_file(tmp_file.name)
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

    ## main

    @f_app.post("/")
    async def main(input_data: UrlInput, api_key: bool = Security(verify_api_key)) -> str:
        return endpoint(input_data.image_url)

    @f_app.post("/upload")
    async def main_upload(image_file: UploadFile, api_key: bool = Security(verify_api_key)) -> str:
        return endpoint(image_file=image_file)

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


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():  # noqa: C901
    return get_app()


## For testing
@app.local_entrypoint()
def main():
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
        files={"image_file": open(DEFAULT_IMG_PATHS[0], "rb")},
        headers={"X-API-Key": api_key},
    )
    assert response.ok, response.status_code


if __name__ == "__main__":
    uvicorn.run(get_app(), reload=True)

# TODO
# - add multiple uploads/urls
# - add user authentication:
#   - save gens and keys to user account
#   - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#       - Only allow authorized users to upload files: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
