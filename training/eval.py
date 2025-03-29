import random
import re
from itertools import chain
from pathlib import Path

import jiwer
import modal
import torch
import yaml
from more_itertools import chunked
from tqdm import tqdm

from utils import (
    APP_NAME,
    BASE_HF_MODEL,
    BASE_QUANT_MODEL,
    CPU,
    DATA_VOL_PATH,
    DEFAULT_USER_PROMPT,
    DPO_HF_MODEL,
    DPO_QUANT_MODEL,
    GPU_IMAGE,
    MAX_NUM_SEQS,
    MEM,
    MINUTES,
    SECRETS,
    SFT_HF_MODEL,
    SFT_QUANT_MODEL,
    SPLITS,
    VOLUME_CONFIG,
    run_model,
)

MAX_SAMPLES = 100

# -----------------------------------------------------------------------------

# Modal
TIMEOUT = 24 * 60 * MINUTES

if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 1

GPU_TYPE = "L40S"
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
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TIMEOUT,
)
def run_model_on_modal(
    img_paths: list[Path],
    model: str,
    quant: bool,
) -> list[str]:
    return run_model(
        img_paths,
        model,
        DEFAULT_USER_PROMPT,
        quant,
        GPU_COUNT,
    )


def main(base: bool, sft: bool, dpo: bool, quant: bool):
    if not base and not sft and not dpo:
        raise ValueError("Must specify at least one of `base`, `sft`, or `dpo`")

    split_cers = {}
    for split in SPLITS:
        json_path = DATA_VOL_PATH / f"sft_{split}.json"
        with open(json_path, "r") as f:
            read_ds = yaml.safe_load(f)

        # Randomly sample MAX_SAMPLES from the dataset
        random_indices = random.sample(range(len(read_ds)), min(MAX_SAMPLES, len(read_ds)))
        img_paths = [read_ds[i]["images"][0] for i in random_indices]
        labels = [read_ds[i]["conversations"][1]["value"] for i in random_indices]

        ## run
        img_batches = list(chunked(img_paths, MAX_NUM_SEQS))
        model = (
            BASE_HF_MODEL
            if base and not quant
            else SFT_HF_MODEL
            if sft and not quant
            else DPO_HF_MODEL
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
                    chain.from_iterable(run_model_on_modal.local(batch, model, quant) for batch in img_batches),
                    desc=split,
                    total=len(img_batches),
                )
            )
        else:
            lst_preds = run_model_on_modal.starmap([(batch, model, quant) for batch in img_batches])
            preds = [item for lst in lst_preds for item in lst]

        split_cers[split] = compute_cer(labels, preds)

    for split, cer in split_cers.items():
        print(f"{split} CER: {cer:.4f}")


@app.function(
    image=GPU_IMAGE,
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
