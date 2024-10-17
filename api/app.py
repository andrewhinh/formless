import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import modal
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from PIL import ImageFile

from utils import (
    DATA_VOLUME,
    DEFAULT_API_KEY,
    DEFAULT_IMG_URL,
    GPU_IMAGE,
    MINUTES,
    NAME,
    VOLUME_CONFIG,
    Colors,
)

parent_path: Path = Path(__file__).parent
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -----------------------------------------------------------------------------

only_download = False  # turn off gpu for initial download
model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
gpu_memory_utilization = 0.90
max_model_len = 8192
max_num_seqs = 1
enforce_eager = True

question = "What is the content of this image?"
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


# Modal
IMAGE = GPU_IMAGE.pip_install(  # add Python dependencies
    "vllm==0.6.2", "term-image==0.7.2", "python-fasthtml==0.6.10", "sqlite-utils==3.18"
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

in_prod = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"

# -----------------------------------------------------------------------------


# Model
@app.cls(
    image=IMAGE,
    gpu=None if only_download else GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=[modal.Secret.from_dotenv(path=parent_path, filename=".env" if in_prod else ".env.dev")],
    timeout=API_TIMEOUT,
    container_idle_timeout=API_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=API_ALLOW_CONCURRENT_INPUTS,
)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    async def download_model(self):
        from huggingface_hub import login, snapshot_download
        from vllm import LLM

        login(token=os.getenv("HF_TOKEN"), new_session=False)
        snapshot_download(
            config["model"],
            ignore_patterns=["*.pt", "*.bin"],
        )

        if config["only_download"]:
            return

        self.llm = LLM(
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
        from fasthtml import common as fh

        db_path = f"/{DATA_VOLUME}/main.db"
        tables = fh.database(db_path).t
        api_keys = tables.api_keys
        if api_key_header in api_keys:
            return True
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    @modal.web_endpoint(method="POST")
    async def infer(self, request: dict, api_key: bool = Security(verify_api_key)) -> str:
        if config["only_download"]:
            return ""

        import requests
        from PIL import Image
        from term_image.image import from_file
        from vllm import SamplingParams

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = request.get("image_url")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        prompt = f"<|image|><|begin_of_text|>{config['question']}"
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

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
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


## For testing
@app.local_entrypoint()
def main(
    twice=True,
):
    import requests

    model = Model()

    response = requests.post(
        model.infer.web_url, json={"image_url": DEFAULT_IMG_URL}, headers={"X-API-Key": DEFAULT_API_KEY}
    )
    assert response.ok, response.status_code

    if twice:
        # second response is faster, because the Function is already running
        response = requests.post(
            model.infer.web_url, json={"image_url": DEFAULT_IMG_URL}, headers={"X-API-Key": DEFAULT_API_KEY}
        )
        assert response.ok, response.status_code
