import os
import time
from pathlib import Path
from uuid import uuid4

import modal

from api.utils import (
    DEPLOY_ALLOW_CONCURRENT_INPUTS,
    DEPLOY_CONTAINER_IDLE_TIMEOUT,
    DEPLOY_TIMEOUT,
    IMAGE,
    NAME,
    VOLUME_CONFIG,
    Colors,
)

only_download = False  # turn off gpu for initial download
model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
gpu_memory_utilization = 0.90
max_model_len = 8192
max_num_seqs = 1
enforce_eager = True

image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
question = "What is the content of this image?"
temperature = 0.2
max_tokens = 128

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
GPU_TYPE = "H100"
GPU_COUNT = 2
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-deploy"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------


# Model
@app.cls(
    image=IMAGE,
    gpu=None if only_download else GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=[modal.Secret.from_dotenv(path=Path(__file__).parent)],
    timeout=DEPLOY_TIMEOUT,
    container_idle_timeout=DEPLOY_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=DEPLOY_ALLOW_CONCURRENT_INPUTS,
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

    @modal.web_endpoint(method="POST")
    async def infer(self, request: dict) -> str:
        if config["only_download"]:
            return ""

        import requests
        from PIL import Image
        from vllm import SamplingParams

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        response = requests.get(request.get("image_url"), stream=True)
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

    response = requests.post(model.infer.web_url, json={"image_url": config["image_url"]})
    assert response.ok, response.status_code

    if twice:
        # second response is faster, because the Function is already running
        response = requests.post(model.infer.web_url, json={"image_url": config["image_url"]})
        assert response.ok, response.status_code
