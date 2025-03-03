"""Official AWQ <-> Qwen2-VL quantize scripts."""

import json
import multiprocessing
import os
import random

import modal
import torch
import torch.nn as nn
from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device
from awq.utils.qwen_vl_utils import process_vision_info
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utils import (
    APP_NAME,
    DATA_VOLUME,
    GPU_IMAGE,
    HF_USERNAME,
    MINUTES,
    PARENT_PATH,
    REGION,
    RUNS_VOLUME,
    SECRETS,
    VOLUME_CONFIG,
)

random.seed(42)

if modal.is_local():
    DATA_VOL_PATH = str(PARENT_PATH / "training" / "artifacts" / "mathwriting-2024")
    RUNS_VOL_PATH = str(PARENT_PATH / "training" / "artifacts" / "runs")
    if not os.path.exists(DATA_VOL_PATH):
        raise Exception(f"""
{DATA_VOL_PATH} does not exist.
""")
else:
    DATA_VOL_PATH = f"/{DATA_VOLUME}"
    RUNS_VOL_PATH = f"/{RUNS_VOLUME}"
CALIBRATION_DATA = f"{DATA_VOL_PATH}/sft_valid.json"
MAX_SAMPLES = 3

# -----------------------------------------------------------------------------

# sft

SFT_PROCESSOR = "Qwen/Qwen2-VL-7B-Instruct"
SFT_MODEL = "qwen2-vl-7b-instruct-lora-sft-merged"
SFT_HF_MODEL = f"{HF_USERNAME}/{SFT_MODEL}"  # pretrained model or ckpt
SFT_QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
SFT_SAVE_PATH = f"{RUNS_VOL_PATH}/{SFT_MODEL}-awq"
SFT_SAVE_HUB = f"{HF_USERNAME}/{SFT_MODEL}-awq"

# -----------------------------------------------------------------------------

# dpo

DPO_PROCESSOR = "Qwen/Qwen2-VL-7B-Instruct"
DPO_MODEL = "qwen2-vl-7b-instruct-lora-dpo-merged"
DPO_HF_MODEL = f"{HF_USERNAME}/{DPO_MODEL}"
DPO_QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
DPO_SAVE_PATH = f"{RUNS_VOL_PATH}/{DPO_MODEL}-awq"
DPO_SAVE_HUB = f"{HF_USERNAME}/{DPO_MODEL}-awq"

# -----------------------------------------------------------------------------


IMAGE = GPU_IMAGE.pip_install(
    "autoawq==0.2.7.post3",
    "torchvision==0.20.1",
)
TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "h100"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

app = modal.App(name=f"{APP_NAME}-quantize")

# -----------------------------------------------------------------------------


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


def cal_data(sample: dict) -> list[dict]:
    for idx, message in enumerate(sample["conversations"]):
        if message["from"] == "human" and "<image>" in message["value"]:
            image_path = sample["images"][0]  # Assuming one image per message
            user_content = [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": message["value"].replace("<image>", "").strip()},
            ]
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": sample["conversations"][idx + 1]["value"]},
            ]
        else:
            return [
                {"role": message["from"], "content": message["value"]},
            ]


def helper(processor, model, quant_config, save_path, save_hub):
    processor = AutoProcessor.from_pretrained(processor)
    model = AutoAWQForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )

    # load calibration data
    with open(CALIBRATION_DATA, "r") as f:
        read_ds = json.load(f)

    cal_ds = list(
        tqdm(
            thread_map(
                cal_data,
                read_ds,
                max_workers=multiprocessing.cpu_count(),
            ),
            total=len(read_ds),
        )
    )
    random.shuffle(cal_ds)
    cal_ds = cal_ds[:MAX_SAMPLES]

    # process the dataset into tensors
    text = processor.apply_chat_template(cal_ds, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(cal_ds)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    # quantize and save
    model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)
    model.model.config.use_cache = model.model.generation_config.use_cache = True
    model.save_quantized(save_path, safetensors=True, shard_size="4GB")
    processor.save_pretrained(save_path)

    # load model from save path and push to hub
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        save_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.push_to_hub(save_hub)
    processor.push_to_hub(save_hub)


def main(sft: bool, dpo: bool):
    if not sft and not dpo:
        raise ValueError("Must specify at least one of `sft` or `dpo`")

    if sft:
        helper(
            SFT_PROCESSOR,
            SFT_HF_MODEL,
            SFT_QUANT_CONFIG,
            SFT_SAVE_PATH,
            SFT_SAVE_HUB,
        )
    if dpo:
        helper(
            DPO_PROCESSOR,
            DPO_HF_MODEL,
            DPO_QUANT_CONFIG,
            DPO_SAVE_PATH,
            DPO_SAVE_HUB,
        )


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    region=REGION,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def run(sft: bool, dpo: bool):
    main(sft, dpo)


@app.local_entrypoint()
def local(sft: bool = False, dpo: bool = False):
    run.remote(sft, dpo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    args = parser.parse_args()
    main(args.sft, args.dpo)
