"""Running official Qwen-VL training script on Modal."""

import os

import modal

from utils import DATA_VOLUME, GPU_IMAGE, MINUTES, NAME, RUNS_VOLUME, SECRETS, VOLUME_CONFIG, _exec_subprocess

# -----------------------------------------------------------------------------

IMAGE = (
    GPU_IMAGE.run_commands(
        [
            "git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git",
            "cd /LLaMA-Factory && pip install -e '.[torch,metrics]'",
        ]
    )
    .copy_local_file("training/dataset_info.json", "/LLaMA-Factory/data/dataset_info.json")
    .copy_local_file("training/ds_z3_config.json", "/LLaMA-Factory/ds_z3_config.json")
    .copy_local_file("training/qwen2vl_full_sft.yaml", "/LLaMA-Factory/qwen2vl_full_sft.yaml")
    .copy_local_file("training/qwen2vl_lora_dpo_train.yaml", "/LLaMA-Factory/qwen2vl_lora_dpo_train.yaml")
    .copy_local_file("training/qwen2vl_lora_dpo_merge.yaml", "/LLaMA-Factory/qwen2vl_lora_dpo_merge.yaml")
    .pip_install(
        "deepspeed==0.15.4",
    )
    .env(
        {
            "FORCE_TORCHRUN": "1",
        }
    )
)
TRAIN_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "H100"
GPU_COUNT = 2
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-train"
app = modal.App(name=APP_NAME)

# -----------------------------------------------------------------------------

with IMAGE.imports():
    from pathlib import Path

    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    def push_to_hub(local_dir: str, model_name: str):
        model = Qwen2VLForConditionalGeneration.from_pretrained(local_dir)
        processor = AutoProcessor.from_pretrained(local_dir)
        api = HfApi()
        user_info = api.whoami(token=os.getenv("HF_TOKEN"))
        username = user_info["name"]
        model.push_to_hub(username + "/" + model_name)
        processor.push_to_hub(username + "/" + model_name)


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TRAIN_TIMEOUT,
)
def run(cls: bool, sft: bool, dpo: bool):
    # run training
    os.chdir("/LLaMA-Factory")
    _exec_subprocess(
        [
            "cp",
            f"/{DATA_VOLUME}/train/data.json",
            "data/data.json",
        ]
    )
    if sft:
        _exec_subprocess(
            [
                "llamafactory-cli",
                "train",
                "qwen2vl_full_sft.yaml",
            ]
        )
        checkpoint_folder = str(
            max(
                list((Path(f"/{RUNS_VOLUME}/qwen2-vl-7b-instruct-full-sft").glob("checkpoint-*"))),
                key=lambda x: int(x.name.split("-")[-1]),
            )
        )
        push_to_hub(checkpoint_folder, "qwen2-vl-7b-instruct-full-sft")
    if dpo:
        _exec_subprocess(
            [
                "llamafactory-cli",
                "train",
                "qwen2vl_lora_dpo_train.yaml",
            ]
        )
        push_to_hub(f"/{RUNS_VOLUME}/qwen2-vl-7b-instruct-lora-dpo", "qwen2-vl-7b-instruct-lora-dpo")
        _exec_subprocess(
            [
                "llamafactory-cli",
                "export",
                "qwen2vl_lora_dpo_merge.yaml",
            ]
        )
        push_to_hub(f"/{RUNS_VOLUME}/qwen2-vl-7b-instruct-lora-dpo-merged", "qwen2-vl-7b-instruct-lora-dpo-merged")

    # TODO: use to create tokens and test model
    # _COMMAND_RE = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")

    # def tokenize_expression(s: str) -> list[str]:
    #     r"""Transform a Latex math string into a list of tokens.

    #     Tokens are strings that are meaningful in the context of Latex
    #     e.g. '1', r'\alpha', r'\frac'.

    #     Args:
    #       s: unicode input string (ex: r"\frac{1}{2}")

    #     Returns
    #     -------
    #       tokens: list of tokens as unicode strings.
    #     """
    #     tokens = []
    #     while s:
    #         if s[0] == "\\":
    #             tokens.append(_COMMAND_RE.match(s).group(0))
    #         else:
    #             tokens.append(s[0])

    #         s = s[len(tokens[-1]) :]

    #     return tokens

    # # Example
    # print(tokenize_expression(r"\frac{\alpha}{2}\begin{matrix}1&0\\0&1\end{matrix}\not\in\mathbb{R}"))
    # ### CER Computation

    # def compute_cer(truth_and_output: list[tuple[str, str]]):
    #     """Computes CER given pairs of ground truth and model output."""

    #     class TokenizeTransform(jiwer.transforms.AbstractTransform):
    #         def process_string(self, s: str):
    #             return tokenize_expression(r"{}".format(s))

    #         def process_list(self, tokens: list[str]):
    #             return [self.process_string(token) for token in tokens]

    #     ground_truth, model_output = zip(*truth_and_output, strict=False)

    #     return jiwer.cer(
    #         truth=list(ground_truth),
    #         hypothesis=list(model_output),
    #         reference_transform=TokenizeTransform(),
    #         hypothesis_transform=TokenizeTransform(),
    #     )

    # # Test data to run compute_cer().
    # # The first element is the model prediction, the second the ground truth.
    # examples = [
    #     (r"\sqrt{2}", r"\sqrt{2}"),  # 0 mistakes, 4 tokens
    #     (r"\frac{1}{2}", r"\frac{i}{2}"),  # 1 mistake, 7 tokens
    #     (r"\alpha^{2}", "a^{2}"),  # 1 mistake, 5 tokens
    #     ("abc", "def"),  # 3 mistakes, 3 tokens
    # ]

    # # 5 mistakes for 19 tokens: 26.3% error rate.
    # print(f"{compute_cer(examples)*100:.1f} %")


@app.local_entrypoint()
def main(cls: bool = False, sft: bool = False, dpo: bool = False):
    if not cls and not sft and not dpo:
        raise ValueError("Must specify at least one of `cls`, `sft`, or `dpo`")
    run.remote(cls, sft, dpo)


# TODO:
# - get better data
# - recreate in pytorch and inc torchtitan
# - replace slow act + fns with custom CUDA kernels
#   - better yet: recreate in C/Cuda
