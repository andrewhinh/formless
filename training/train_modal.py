import subprocess

import modal
from ft.utils import (
    CPU,
    IMAGE,
    PREFIX_PATH,
    TIMEOUT,
    TRAIN_SCRIPT_PATH,
    VOLUME_CONFIG,
)

# Modal
GPU_TYPE = "H100"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "train_model"
app = modal.App(name=APP_NAME)


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


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes=VOLUME_CONFIG,
    cpu=CPU,
)
def run():
    command = (
        f"torchrun --standalone --nproc_per_node={GPU_COUNT} {TRAIN_SCRIPT_PATH} --no_local"
        if GPU_COUNT > 1
        else f"python {TRAIN_SCRIPT_PATH} --no_local"
    )
    _exec_subprocess(command.split())


@app.local_entrypoint()
def main():
    run.remote()
