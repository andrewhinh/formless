"""Running official Qwen-VL training script on Modal."""

import os

import modal
import yaml

from utils import DATA_VOLUME, GPU_IMAGE, MINUTES, NAME, RUNS_VOLUME, SECRETS, VOLUME_CONFIG

# -----------------------------------------------------------------------------

PROCESSOR = "Qwen/Qwen2-VL-7B-Instruct"
MODEL = "andrewhinh/qwen2-vl-7b-instruct"  # pretrained model or ckpt
CALIBRATION_DATA = f"/{DATA_VOLUME}/train/data.json"
QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
SAVE_PATH = f"{RUNS_VOLUME}/qwen2-vl-7b-instruct-awq"
SAVE_HUB = "andrewhinh/qwen2-vl-7b-instruct-awq"

# -----------------------------------------------------------------------------

IMAGE = GPU_IMAGE.pip_install(
    "autoawq==0.2.7.post3",
    "torchvision==0.20.1",
    "ninja==1.11.1",  # required to build flash-attn
    "packaging==23.1",  # required to build flash-attn
    "wheel==0.41.2",  # required to build flash-attn
    "torch==2.5.1",  # required to build flash-attn
).run_commands(  # add flash-attn
    "pip install flash-attn==2.7.2.post1 --no-build-isolation"
)
QUANTIZE_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "h100"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-quantize"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------

with IMAGE.imports():
    import torch
    import torch.nn as nn
    from awq import AutoAWQForCausalLM
    from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device
    from awq.utils.qwen_vl_utils import process_vision_info
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    # We define our own quantizer by extending the AwqQuantizer.
    # The main difference is in how the samples are processed when
    # the quantization process initialized.
    class Qwen2VLAwqQuantizer(AwqQuantizer):
        def init_quant(self, n_samples=None, max_seq_len=None):  # noqa: C901
            modules = self.awq_model.get_model_layers(self.model)
            samples = self.calib_data

            inps = []
            layer_kwargs = {}

            best_device = get_best_device()
            modules[0] = modules[0].to(best_device)
            self.awq_model.move_embed(self.model, best_device)

            # get input and kwargs to layer 0
            # with_kwargs is only supported in PyTorch 2.0
            # use this Catcher hack for now
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module

                def forward(self, *args, **kwargs):
                    # assume first input to forward is hidden states
                    if len(args) > 0:
                        hidden_states = args[0]
                        del args
                    else:
                        first_key = list(kwargs.keys())[0]
                        hidden_states = kwargs.pop(first_key)

                    inps.append(hidden_states)
                    layer_kwargs.update(kwargs)
                    raise ValueError  # early exit to break later inference

            def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
                def get_device(obj: torch.Tensor | nn.Module):
                    if isinstance(obj, torch.Tensor):
                        return obj.device
                    return next(obj.parameters()).device

                if get_device(obj) != device:
                    obj = obj.to(device)
                return obj

            # patch layer 0 to catch input and kwargs
            modules[0] = Catcher(modules[0])
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, best_device)
            try:
                self.model(**samples)
            except ValueError:  # work with early exit
                pass
            finally:
                for k, v in samples.items():
                    if isinstance(v, (torch.Tensor, nn.Module)):
                        samples[k] = move_to_device(v, "cpu")
            modules[0] = modules[0].module  # restore

            del samples
            inps = inps[0]

            modules[0] = modules[0].cpu()
            self.awq_model.move_embed(self.model, "cpu")

            clear_memory()

            return modules, layer_kwargs, inps


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=QUANTIZE_TIMEOUT,
)
def run():
    processor = Qwen2VLProcessor.from_pretrained(PROCESSOR)
    model = AutoAWQForCausalLM.from_pretrained(MODEL, attn_implementation="flash_attention_2")

    # load calibration data
    with open(CALIBRATION_DATA, "r") as f:
        train_ds = yaml.safe_load(f)
    cal_ds = []
    for sample in train_ds:
        for idx, message in enumerate(sample["messages"]):
            if message["role"] == "user" and "<image>" in message["content"]:
                image_path = sample["images"][0]  # Assuming one image per message
                user_content = [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": message["content"].replace("<image>", "").strip()},
                ]
                cal_ds.append(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": sample["messages"][idx + 1]["content"]},
                    ]
                )
            else:
                cal_ds.append(
                    [
                        {"role": message["role"], "content": message["content"]},
                    ]
                )

    # process the dataset into tensors
    text = processor.apply_chat_template(cal_ds, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(cal_ds)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    # quantize and save
    model.quantize(calib_data=inputs, quant_config=QUANT_CONFIG, quantizer_cls=Qwen2VLAwqQuantizer)
    model.model.config.use_cache = model.model.generation_config.use_cache = True
    model.save_quantized(SAVE_PATH, safetensors=True, shard_size="4GB")
    processor.save_pretrained(SAVE_PATH)

    # load model from save path and push to hub
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SAVE_PATH,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.push_to_hub(SAVE_HUB)
    processor.push_to_hub(SAVE_HUB)


@app.local_entrypoint()
def main():
    run.remote()
