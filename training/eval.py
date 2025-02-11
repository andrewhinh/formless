import base64
import os
import re
from itertools import chain
from pathlib import Path

import jiwer
import modal
import torch
import yaml
from huggingface_hub import login
from more_itertools import chunked
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    APP_NAME,
    DATA_VOLUME,
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    GPU_IMAGE,
    HF_USERNAME,
    MINUTES,
    PARENT_PATH,
    SECRETS,
    VOLUME_CONFIG,
)

# -----------------------------------------------------------------------------

# data
if modal.is_local():
    DATA_VOL_PATH = str(PARENT_PATH / "training" / "artifacts" / "mathwriting-2024")
    if not os.path.exists(DATA_VOL_PATH):
        raise Exception(f"""
{DATA_VOL_PATH} does not exist.
""")
else:
    DATA_VOL_PATH = f"/{DATA_VOLUME}"
SPLITS = ["train", "valid", "test"]
SFT_TRAIN_JSON = Path(f"{DATA_VOL_PATH}/sft_train.json")
SFT_VAL_JSON = Path(f"{DATA_VOL_PATH}/sft_valid.json")
SFT_TEST_JSON = Path(f"{DATA_VOL_PATH}/sft_test.json")

# vlm config

TOKENIZER = "Qwen/Qwen2-VL-7B-Instruct"
BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
BASE_QUANT_MODEL = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
SFT_MODEL = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-sft-merged"
SFT_QUANT_MODEL = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-sft-merged-awq"
DPO_MODEL = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-dpo-merged"
DPO_QUANT_MODEL = f"{HF_USERNAME}/{APP_NAME}-qwen2-vl-7b-instruct-lora-dpo-merged-awq"

KV_CACHE_DTYPE = None  # "fp8_e5m2"
ENFORCE_EAGER = False
MAX_NUM_SEQS = 8
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TEMPERATURE = 0.0
TOP_P = 0.001
REPEATION_PENALTY = 1.05
STOP_TOKEN_IDS = []
MAX_MODEL_LEN = 16384

# LATEX_GRAMMER = """
# start: math_expr

# math_expr: inline_math | display_math

# inline_math: "$" expr "$"
# display_math: BEGIN "{" ENV "}" expr END "{" ENV "}"

# ?expr: term
#      | expr operator term   -> binop

# ?term: NUMBER
#      | VARIABLE
#      | function
#      | "{" expr "}"
#      | FRAC "{" expr "}" "{" expr "}"

# function: SIN "(" expr ")"
#         | COS "(" expr ")"
#         | TAN "(" expr ")"
#         | LOG "(" expr ")"
#         | EXP "(" expr ")"
#         | SQRT "{" expr "}"

# operator: "+" | "-" | "*" | "/" | "=" | "^"

# BEGIN: "\\begin"
# END: "\\end"
# FRAC: "\\frac"
# SIN: "\\sin"
# COS: "\\cos"
# TAN: "\\tan"
# LOG: "\\log"
# EXP: "\\exp"
# SQRT: "\\sqrt"

# ENV: /[a-zA-Z]+/

# %import common.NUMBER
# %import common.CNAME -> VARIABLE
# %import common.WS
# %ignore WS
# """

# -----------------------------------------------------------------------------


## container startup fn
def download_models():
    from huggingface_hub import snapshot_download

    login(token=os.getenv("HF_TOKEN"), new_session=False)

    for model in [TOKENIZER, BASE_MODEL, BASE_QUANT_MODEL, SFT_MODEL, SFT_QUANT_MODEL, DPO_MODEL, DPO_QUANT_MODEL]:
        if not os.path.exists(model):
            snapshot_download(
                model,
                ignore_patterns=["*.pt", "*.bin"],
            )
        else:  # check if preprocessor_config.json was successfully copied; if not, do so
            if not os.path.exists(f"{model}/preprocessor_config.json"):
                tok_path = snapshot_download(
                    model,
                    ignore_patterns=["*.pt", "*.bin"],
                )
                os.rename(f"{tok_path}/preprocessor_config.json", f"{model}/preprocessor_config.json")


# Modal
IMAGE = GPU_IMAGE.pip_install("jiwer==3.1.0", "more-itertools==10.6.0").run_function(
    download_models,
    secrets=SECRETS,
    volumes=VOLUME_CONFIG,
)
TIMEOUT = 24 * 60 * MINUTES

if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 8

GPU_TYPE = "h100"
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

app = modal.App(name=f"{APP_NAME}-eval")

# -----------------------------------------------------------------------------


def tokenize_expression(s: str) -> list[str]:
    r"""Transform a Latex math string into a list of tokens.

    Tokens are strings that are meaningful in the context of Latex
    e.g. '1', r'\alpha', r'\frac'.

    Args:
        s: unicode input string (ex: r"\frac{1}{2}")

    Returns
    -------
        tokens: list of tokens as unicode strings.
    """
    _COMMAND_RE = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")
    tokens = []
    while s:
        if s[0] == "\\":
            match = _COMMAND_RE.match(s)
            if match:
                tokens.append(match.group(0))
                s = s[len(tokens[-1]) :]
            else:
                tokens.append(s[0])
                s = s[1:]
        else:
            tokens.append(s[0])
            s = s[1:]

    return tokens


def compute_cer(gt: list[str], output: list[str]) -> float:
    """Computes CER given pairs of ground truth and model output."""

    class TokenizeTransform(jiwer.transforms.AbstractTransform):
        def process_string(self, s: str):
            return tokenize_expression(r"{}".format(s))

        def process_list(self, tokens: list[str]):
            return [self.process_string(token) for token in tokens]

    return jiwer.cer(
        truth=gt,
        hypothesis=output,
        reference_transform=TokenizeTransform(),
        hypothesis_transform=TokenizeTransform(),
    )


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def run_model(img_paths: list[Path], model: str, quant: bool) -> list[str]:
    conversations = []
    for img_path in img_paths:
        with open(img_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{base64_img}"
        conversations.append(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_QUESTION},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                },
            ]
        )

    global quantization
    global llm
    global sampling_params
    # load pretrained vlm if not already loaded
    if "quantization" not in globals():
        quantization = "awq_marlin" if quant else None
    if "llm" not in globals():
        llm = LLM(
            model=model,
            enforce_eager=ENFORCE_EAGER,
            max_num_seqs=MAX_NUM_SEQS,
            tensor_parallel_size=GPU_COUNT,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            mm_processor_kwargs={
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            },
            **{k: v for k, v in [("quantization", quantization), ("kv_cache_dtype", KV_CACHE_DTYPE)] if v is not None},
        )
    if "sampling_params" not in globals():
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPEATION_PENALTY,
            stop_token_ids=STOP_TOKEN_IDS,
            guided_decoding=None,  # GuidedDecodingParams(grammar=LATEX_GRAMMER, backend="outlines"),
        )
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    preds = [out.outputs[0].text.strip() for out in outputs]
    return preds


def main(base: bool, sft: bool, dpo: bool, quant: bool):
    if not base and not sft and not dpo:
        raise ValueError("Must specify at least one of `base`, `sft`, or `dpo`")

    split_cers = {}
    for split in SPLITS:
        json_path = SFT_TRAIN_JSON if split == "train" else SFT_VAL_JSON if split == "valid" else SFT_TEST_JSON
        with open(json_path, "r") as f:
            read_ds = yaml.safe_load(f)
        img_paths = [sample["images"][0] for sample in read_ds]
        labels = [sample["conversations"][1]["value"] for sample in read_ds]

        ## run
        img_batches = list(chunked(img_paths, MAX_NUM_SEQS))
        model = (
            BASE_MODEL
            if base and not quant
            else SFT_MODEL
            if sft and not quant
            else DPO_MODEL
            if dpo and not quant
            else BASE_QUANT_MODEL
            if base and quant
            else SFT_QUANT_MODEL
            if sft and quant
            else DPO_QUANT_MODEL
            if dpo and quant
            else None
        )
        if modal.is_local():
            preds = list(
                tqdm(
                    chain.from_iterable(run_model.local(batch, model, quant) for batch in img_batches),
                    desc=split,
                    total=len(img_batches),
                )
            )
        else:
            lst_preds = run_model.starmap([(batch, model, quant) for batch in img_batches])
            preds = [item for lst in lst_preds for item in lst]

        split_cers[split] = compute_cer(labels, preds)

    for split, cer in split_cers.items():
        print(f"{split} CER: {cer:.4f}")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def run(base: bool, sft: bool, dpo: bool, quant: bool):
    main(base, sft, dpo, quant)


@app.local_entrypoint()
def local(base: bool = False, sft: bool = False, dpo: bool = False, quant: bool = False):
    run.remote(base, sft, dpo, quant)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    parser.add_argument("--quant", action="store_true")
    args = parser.parse_args()
    main(args.base, args.sft, args.dpo, args.quant)
