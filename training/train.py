"""Official Timm/Qwen2-VL training scripts."""

import os
import sys
from pathlib import Path

# since file must be run inside training/, and utils is in project root
project_root = os.path.abspath(Path(__file__).parent.parent)
sys.path.append(project_root)


import importlib
import json
import logging
import random
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import modal
import torch
import torch.nn as nn
import torchvision.utils
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from timm import utils
from timm.data import AugMixDataset, FastCollateMixup, Mixup, create_dataset, create_loader, resolve_data_config
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import (
    create_model,
    load_checkpoint,
    model_parameters,
    push_to_hf_hub,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utils import DATA_VOLUME, GPU_IMAGE, IN_PROD, MINUTES, NAME, RUNS_VOLUME, SECRETS, VOLUME_CONFIG, _exec_subprocess

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if torch.cuda.amp.autocast is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion  # noqa: F401

    has_functorch = True
except ImportError:
    has_functorch = False

has_compile = hasattr(torch, "compile")

_logger = logging.getLogger("train")

PARENT_PATH = Path(__file__).parent
ARTIFACT_PATH = PARENT_PATH / "artifacts"

# setup
seed = 42  # random seed (default: 42)
random.seed(seed)
load_dotenv(".env" if IN_PROD else ".env.dev")

api = HfApi()
user_info = api.whoami(token=os.getenv("HF_TOKEN"))
HF_USERNAME = user_info["name"]


# -----------------------------------------------------------------------------

# cls

## dataset
CLASSES = ["1", "2", "3"]
data_dir = ARTIFACT_PATH / "mathwriting-2024"  # path to dataset (root dir)
dataset = ""  # dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)
train_split = "train"  # dataset train split (default: train)
val_split = "valid"  # dataset validation split (default: validation)
input_img_mode = None  # Dataset image conversion mode for input images.
input_key = None  # Dataset key for input images.
target_key = None  # Dataset key for target labels.

## model
model = "resnet152"  # Name of model to train (default: "resnet50")
pretrained = True  # Start with pretrained version of specified network (if avail)
pretrained_path = None  # Load this checkpoint as if they were the pretrained weights (with adaptation).
initial_checkpoint = ""  # Load this checkpoint into model after initialization (default: none)
resume = ""  # Resume full model and optimizer state from checkpoint (default: none)
no_resume_opt = False  # prevent resume of optimizer state when resuming model
num_classes = len(CLASSES)  # number of label classes (Model default if None)
gp = None  # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.
img_size = None  # Image size (default: None => model default)
in_chans = None  # Image input channels (default: None => 3)
input_size = None  # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
crop_pct = None  # Input image center crop percent (for validation only)
mean = None  # Override mean pixel value of dataset
std = None  # Override std deviation of dataset
interpolation = ""  # Image resize interpolation type (overrides model)
batch_size = 128  # Input batch size for training (default: 128)
validation_batch_size = None  # Validation batch size override (default: None)
channels_last = False  # Use channels_last memory layout
fuser = ""  # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')
grad_accum_steps = 1  # The number of steps to accumulate gradients (default: 1)
grad_checkpointing = False  # Enable gradient checkpointing through model blocks/stages
fast_norm = False  # enable experimental fast-norm
model_kwargs = {}  # Additional model keyword arguments
head_init_scale = None  # Head initialization scale
head_init_bias = None  # Head initialization bias value

## scripting / codegen
torchscript = False  # torch.jit.script the full model
torchcompile = None  # Enable compilation w/ specified backend (default: "inductor")

## device & distributed
device = "cpu"  # Device (accelerator) to use.
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
ddp_backend = "nccl"  # Distributed backend (default: "nccl")
amp = True  # use NVIDIA Apex AMP or Native AMP for mixed precision training
amp_dtype = "bfloat16"  # lower precision AMP dtype (default: float16)
amp_impl = "native"  # AMP impl to use, "native" or "apex" (default: native)
no_ddp_bb = False  # Force broadcast buffers for native DDP to off.
synchronize_step = False  # torch.cuda.synchronize() end of each step
device_modules = None  # Python imports for device backend modules.

## optimizer
opt = "adamw"  # Optimizer (default: "sgd")
opt_eps = None  # Optimizer Epsilon (default: None, use opt default)
opt_betas = None  # Optimizer Betas (default: None, use opt default)
momentum = 0.9  # Optimizer momentum (default: 0.9)
weight_decay = 1e-4  # weight decay (default: 2e-5)
clip_grad = 1.0  # Clip gradient norm (default: None, no clipping)
clip_mode = "norm"  # Gradient clipping mode. One of ("norm", "value", "agc")
layer_decay = None  # layer-wise learning rate decay (default: None)
opt_kwargs = {}  # Additional optimizer keyword arguments

## lr schedule
sched = "cosine"  # LR scheduler (default: "cosine")
sched_on_updates = False  # Apply LR scheduler step on update instead of epoch end.
lr = 3e-4  # learning rate, overrides lr-base if set (default: None)
lr_base = 0.1  # base learning rate: lr = lr_base * global_batch_size / base_size
lr_base_size = 256  # base learning rate batch size (divisor, default: 256).
lr_base_scale = ""  # base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)
lr_noise = None  # learning rate noise on/off epoch percentages
lr_noise_pct = 0.67  # learning rate noise limit percent (default: 0.67)
lr_noise_std = 1.0  # learning rate noise std-dev (default: 1.0)
lr_cycle_mul = 1.0  # learning rate cycle len multiplier (default: 1.0)
lr_cycle_decay = 0.5  # amount to decay each learning rate cycle (default: 0.5)
lr_cycle_limit = 1  # learning rate cycle limit, cycles enabled if > 1
lr_k_decay = 1.0  # learning rate k-decay for cosine/poly (default: 1.0)
warmup_lr = 1e-5  # warmup learning rate (default: 1e-5)
min_lr = 0  # lower lr bound for cyclic schedulers that hit 0 (default: 0)
epochs = 5  # number of epochs to train (default: 300)
epoch_repeats = 0.0  # epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
start_epoch = None  # manual epoch number (useful on restarts)
decay_milestones = [90, 180, 270]  # list of decay epoch indices for multistep lr. must be increasing
decay_epochs = 90  # epoch interval to decay LR
warmup_epochs = 5  # epochs to warmup LR, if scheduler supports
warmup_prefix = False  # Exclude warmup period from decay schedule.
cooldown_epochs = 0  # epochs to cooldown LR at min_lr, after cyclic schedule ends
patience_epochs = 10  # patience epochs for Plateau LR scheduler (default: 10)
decay_rate = 0.1  # LR decay rate (default: 0.1)

## augmentation & regularization
no_aug = False  # Disable all training augmentation, override other train aug args
train_crop_mode = None  # Crop-mode in train
scale = [0.08, 1.0]  # Random resize scale (default: 0.08 1.0)
ratio = [0.75, 1.33]  # Random resize aspect ratio (default: 0.75 1.33)
hflip = 0.5  # Horizontal flip training aug probability
vflip = 0.0  # Vertical flip training aug probability
color_jitter = 0.4  # Color jitter factor (default: 0.4)
color_jitter_prob = None  # Probability of applying any color jitter.
grayscale_prob = None  # Probability of applying random grayscale conversion.
gaussian_blur_prob = None  # Probability of applying gaussian blur.
aa = None  # Use AutoAugment policy. "v0" or "original". (default: None)
aug_repeats = 0  # Number of augmentation repetitions (distributed training only) (default: 0)
aug_splits = 0  # Number of augmentation splits (default: 0, valid: 0 or >=2)
jsd_loss = False  # Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
bce_loss = False  # Enable BCE loss w/ Mixup/CutMix use.
bce_sum = False  # Sum over classes when using BCE loss.
bce_target_thresh = None  # Threshold for binarizing softened BCE targets (default: None, disabled).
bce_pos_weight = None  # Positive weighting for BCE loss.
reprob = 0.0  # Random erase prob (default: 0.)
remode = "pixel"  # Random erase mode (default: "pixel")
recount = 1  # Random erase count (default: 1)
resplit = False  # Do not random erase first (clean) augmentation split
mixup = 0.0  # mixup alpha, mixup enabled if > 0. (default: 0.)
cutmix = 0.0  # cutmix alpha, cutmix enabled if > 0. (default: 0.)
cutmix_minmax = None  # cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
mixup_prob = 1.0  # Probability of performing mixup or cutmix when either/both is enabled
mixup_switch_prob = 0.5  # Probability of switching to cutmix when both mixup and cutmix enabled
mixup_mode = "batch"  # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
mixup_off_epoch = 0  # Turn off mixup after this epoch, disabled if 0 (default: 0)
smoothing = 0.1  # Label smoothing (default: 0.1)
train_interpolation = "random"  # Training interpolation (random, bilinear, bicubic default: "random")
drop = 0.0  # Dropout rate (default: 0.)
drop_connect = None  # Drop connect rate, DEPRECATED, use drop-path (default: None)
drop_path = None  # Drop path rate (default: None)
drop_block = None  # Drop block rate (default: None)

## batch norm parameters (only works with gen_efficientnet based models currently)
bn_momentum = None  # BatchNorm momentum override (if not None)
bn_eps = None  # BatchNorm epsilon override (if not None)
sync_bn = True  # Enable NVIDIA Apex or Torch synchronized BatchNorm.
dist_bn = "reduce"  # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
split_bn = False  # Enable separate BN layers per augmentation split.

## model exp moving avg
model_ema = True  # Enable tracking moving average of model weights.
model_ema_force_cpu = False  # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.
model_ema_decay = 0.9998  # Decay factor for model weights moving average (default: 0.9998)
model_ema_warmup = False  # Enable warmup for model EMA decay.

## misc
worker_seeding = "all"  # worker seed mode (default: all)
log_interval = 1  # how many batches to wait before logging training status
recovery_interval = 0  # how many batches to wait before writing recovery checkpoint
checkpoint_hist = 1  # number of checkpoints to keep (default: 10)
workers = os.cpu_count() // 2 + 1  # how many training processes to use (default: 4)
save_images = False  # save images of input batches every log interval for debugging
pin_mem = False  # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
no_prefetcher = False  # disable fast prefetcher
output = ARTIFACT_PATH / "runs" / "cls"  # path to output folder (default: none, current dir)
experiment = ""  # name of train experiment, name of sub-folder for output
eval_metric = "top1"  # Best metric (default: "top1")
tta = 0  # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
use_multi_epochs_loader = False  # use the multi-epochs-loader to save time at the beginning of every epoch
log_wandb = True  # log training and validation metrics to wandb
model_hub_name = "-".join([safe_model_name(model), "cls"])  # name of the model when pushed to the HF hub

## log to hf
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, Path, type(None)))
]
config = {k: globals()[k] for k in config_keys if not k.isupper()}  # will be useful for logging
config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}  # since Path not serializable

# -----------------------------------------------------------------------------

# Modal
TRAIN_REPO_PATH = PARENT_PATH / "LLaMA-Factory" if modal.is_local() else "/LLaMA-Factory"
IMAGE = (
    GPU_IMAGE.run_commands(
        [
            "git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git",
            "cd /LLaMA-Factory && pip install -e '.[torch,metrics]'",
        ]
    )
    .pip_install("deepspeed==0.15.4", "timm==1.0.14")
    .env(
        {
            "FORCE_TORCHRUN": "1",
        }
    )
)
TRAIN_TIMEOUT = 24 * 60 * MINUTES

GPU_TYPE = "H100"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = f"{NAME}-train"
app = modal.App(name=APP_NAME)


## dataset_info.json
SFT_DATA = "sft_train.json"
DPO_DATA = "dpo_train.json"
dataset_info = {
    "sft": {
        "file_name": f"{TRAIN_REPO_PATH}/data/{SFT_DATA}",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
    },
    "dpo": {
        "file_name": f"{TRAIN_REPO_PATH}/data/{DPO_DATA}",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected", "images": "images"},
    },
}
with open(f"{TRAIN_REPO_PATH}/data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=4)

## ds_z3_config.json
ds_z3_config = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": True,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
}
with open(f"{TRAIN_REPO_PATH}/ds_z3_config.json", "w") as f:
    json.dump(ds_z3_config, f, indent=4)

if modal.is_local():
    DATA_VOL_PATH = str(ARTIFACT_PATH / "mathwriting-2024")
    RUNS_VOL_PATH = str(ARTIFACT_PATH / "runs")
    if not os.path.exists(DATA_VOL_PATH):
        raise Exception(f"""
{DATA_VOL_PATH} does not exist.
""")
else:
    DATA_VOL_PATH = f"/{DATA_VOLUME}"
    RUNS_VOL_PATH = f"/{RUNS_VOLUME}"


# -----------------------------------------------------------------------------

# sft

BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
SFT_MODEL = "qwen2-vl-7b-instruct-lora-sft"
SFT_YAML = "qwen2vl_lora_sft_train.yaml"
SFT_MERGE_YAML = "qwenvl_lora_sft_merge.yaml"

sft_config = {
    ### model
    "model_name_or_path": BASE_MODEL,
    "image_resolution": 262144,
    "video_resolution": 16384,
    "trust_remote_code": True,
    ### method
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_target": "all",
    ### dataset
    "dataset": "sft",
    "template": "qwen2_vl",
    "cutoff_len": 2048,
    "max_samples": 1000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 16,  # 16 = max
    ### output
    "output_dir": f"{RUNS_VOL_PATH}/{SFT_MODEL}",
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "include_effective_tokens_per_second": True,
    ### train
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1.0e-4,
    "num_train_epochs": 5.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "ddp_timeout": 180000000,
    ### eval
    "val_size": 0.1,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 500,
}
with open(f"{TRAIN_REPO_PATH}/{SFT_YAML}", "w") as f:
    yaml.dump(sft_config, f)

sft_merge_config = {
    ### model
    "model_name_or_path": BASE_MODEL,
    "adapter_name_or_path": f"{RUNS_VOL_PATH}/{SFT_MODEL}",
    "template": "qwen2_vl",
    "finetuning_type": "lora",
    "trust_remote_code": True,
    ### export
    "export_dir": f"{RUNS_VOL_PATH}/{SFT_MODEL}-merged",
    "export_size": 2,
    "export_device": "cpu",
    "export_legacy_format": False,
}  ## Note: DO NOT use quantized model or quantization_bit when merging lora adapters
with open(f"{TRAIN_REPO_PATH}/{SFT_MERGE_YAML}", "w") as f:
    yaml.dump(sft_merge_config, f)


# -----------------------------------------------------------------------------

# dpo

DPO_MODEL = "qwen2-vl-7b-instruct-lora-dpo"
DPO_YAML = "qwen2vl_lora_dpo_train.yaml"
DPO_MERGE_YAML = "qwen2vl_lora_dpo_merge.yaml"

dpo_train_config = {
    ### model
    "model_name_or_path": f"{HF_USERNAME}/{SFT_MODEL}-merged",
    "image_resolution": 262144,
    "video_resolution": 16384,
    "trust_remote_code": True,
    ### method
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_target": "all",
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",  # choices: [sigmoid (dpo), orpo, simpo]
    ### dataset
    "dataset": "dpo",
    "template": "qwen2_vl",
    "cutoff_len": 1024,
    "max_samples": 1000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 16,  # 16 = max
    ### output
    "output_dir": f"{RUNS_VOL_PATH}/{DPO_MODEL}",
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "include_effective_tokens_per_second": True,
    ### train
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5.0e-6,
    "num_train_epochs": 5.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "ddp_timeout": 180000000,
    ### eval
    "val_size": 0.1,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 500,
}
with open(f"{TRAIN_REPO_PATH}/{DPO_YAML}", "w") as f:
    yaml.dump(dpo_train_config, f)

dpo_merge_config = {
    ### model
    "model_name_or_path": BASE_MODEL,
    "adapter_name_or_path": f"{RUNS_VOL_PATH}/{DPO_MODEL}",
    "template": "qwen2_vl",
    "finetuning_type": "lora",
    "trust_remote_code": True,
    ### export
    "export_dir": f"{RUNS_VOL_PATH}/{DPO_MODEL}-merged",
    "export_size": 2,
    "export_device": "cpu",
    "export_legacy_format": False,
}  ## Note: DO NOT use quantized model or quantization_bit when merging lora adapters
with open(f"{TRAIN_REPO_PATH}/{DPO_MERGE_YAML}", "w") as f:
    yaml.dump(dpo_merge_config, f)

# -----------------------------------------------------------------------------

# helpers


def run_cls():  # noqa: C901
    utils.setup_default_logging()

    if config["device_modules"]:
        for module in config["device_modules"]:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    prefetcher = not config["no_prefetcher"]
    config["grad_accum_steps"] = max(1, config["grad_accum_steps"])

    distributed = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if distributed:
        init_process_group(backend=config["ddp_backend"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        config["device"] = f"cuda:{local_rank}"
        torch.cuda.set_device(config["device"])
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config["grad_accum_steps"] % world_size == 0
        config["grad_accum_steps"] //= world_size
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {rank}, total {world_size}, device {config['device']}."
        )
    else:
        rank = 0
        world_size = 1
        _logger.info(f"Training with a single process on 1 device ({config['device']}).")
    assert rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if config["amp"]:
        if config["amp_impl"] == "apex":
            assert config["has_apex"], "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert config["amp_dtype"] == "float16"
        else:
            assert config["has_native_amp"], "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert config["amp_dtype"] in ("float16", "bfloat16")
        if config["amp_dtype"] == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(config["seed"], rank)

    if config["fuser"]:
        utils.set_jit_fuser(config["fuser"])
    if config["fast_norm"]:
        set_fast_norm()

    in_chans = 3
    if in_chans is not None:
        in_chans = in_chans
    elif config["input_size"] is not None:
        in_chans = config["input_size"][0]

    factory_kwargs = {}
    if config["pretrained_path"]:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs["pretrained_cfg_overlay"] = {
            "file": config["pretrained_path"],
            "num_classes": -1,  # force head adaptation
        }

    model = create_model(
        config["model"],
        pretrained=config["pretrained"],
        in_chans=in_chans,
        num_classes=config["num_classes"],
        drop_rate=config["drop"],
        drop_path_rate=config["drop_path"],
        drop_block_rate=config["drop_block"],
        global_pool=config["gp"],
        bn_momentum=config["bn_momentum"],
        bn_eps=config["bn_eps"],
        scriptable=config["torchscript"],
        checkpoint_path=config["initial_checkpoint"],
        **factory_kwargs,
        **config["model_kwargs"],
    )
    if config["head_init_scale"] is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(config["head_init_scale"])
            model.get_classifier().bias.mul_(config["head_init_scale"])
    if config["head_init_bias"] is not None:
        nn.init.constant_(model.get_classifier().bias, config["head_init_bias"])

    if config["num_classes"] is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        config["num_classes"] = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if config["grad_checkpointing"]:
        model.set_grad_checkpointing(enable=True)

    if rank == 0:
        _logger.info(
            f"Model {safe_model_name(config['model'])} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )
    data_config = resolve_data_config(config, model=model, verbose=rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if config["aug_splits"] > 0:
        assert config["aug_splits"] > 1, "A split of 1 makes no sense"
        num_aug_splits = config["aug_splits"]

    # enable split bn (separate bn stats per batch-portion)
    if config["split_bn"]:
        assert num_aug_splits > 1 or config["resplit"]
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=config["device"])
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if distributed and config["sync_bn"]:
        config["dist_bn"] = ""  # disable dist_bn when sync BN active
        assert not config["split_bn"]
        if config["has_apex"] and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if config["torchscript"]:
        assert not config["torchcompile"]
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not config["sync_bn"], "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    if not config["lr"]:
        global_batch_size = config["batch_size"] * world_size * config["grad_accum_steps"]
        batch_ratio = global_batch_size / config["lr_base_size"]
        if not config["lr_base_scale"]:
            on = config["opt"].lower()
            config["lr_base_scale"] = "sqrt" if any(o in on for o in ("ada", "lamb")) else "linear"
        if config["lr_base_scale"] == "sqrt":
            batch_ratio = batch_ratio**0.5
        config["lr"] = config["lr_base"] * batch_ratio
        if rank == 0:
            _logger.info(
                f"Learning rate ({config['lr']}) calculated from base learning rate ({config['lr_base']}) "
                f"and effective global batch size ({global_batch_size}) with {config['lr_base_scale']} scaling."
            )

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(SimpleNamespace(**config)), **config["opt_kwargs"])

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert torch.device(config["device"]).type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        try:
            amp_autocast = partial(torch.autocast, device_type=torch.device(config["device"]).type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert torch.device(config["device"]).type == "cuda"
            amp_autocast = torch.cuda.amp.autocast
        if torch.device(config["device"]).type == "cuda" and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if config["resume"]:
        resume_epoch = resume_checkpoint(
            model,
            config["resume"],
            optimizer=None if config["no_resume_opt"] else optimizer,
            loss_scaler=None if config["no_resume_opt"] else loss_scaler,
            log_info=rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if config["model_ema"]:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=config["model_ema_decay"],
            use_warmup=config["model_ema_warmup"],
            device="cpu" if config["model_ema_force_cpu"] else None,
        )
        if config["resume"]:
            load_checkpoint(model_ema.module, config["resume"], use_ema=True)
        if config["torchcompile"]:
            model_ema = torch.compile(model_ema, backend=config["torchcompile"])

    # setup distributed training
    if distributed:
        if config["has_apex"] and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[config["device"]], broadcast_buffers=not config["no_ddp_bb"])
        # NOTE: EMA model does not need to be wrapped by DDP

    if config["torchcompile"]:
        # torch compile should be done after DDP
        assert config[
            "has_compile"
        ], "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        model = torch.compile(model, backend=config["torchcompile"])

    # create the train and eval datasets
    if config["input_img_mode"] is None:
        config["input_img_mode"] = "RGB" if data_config["input_size"][0] == 3 else "L"
    else:
        config["input_img_mode"] = config["input_img_mode"]

    dataset_train = create_dataset(
        config["dataset"],
        root=config["data_dir"],
        split=config["train_split"],
        is_training=True,
        batch_size=config["batch_size"],
        seed=config["seed"],
        repeats=config["epoch_repeats"],
        input_img_mode=config["input_img_mode"],
        input_key=config["input_key"],
        target_key=config["target_key"],
    )

    if config["val_split"]:
        dataset_eval = create_dataset(
            config["dataset"],
            root=config["data_dir"],
            split=config["val_split"],
            is_training=False,
            batch_size=config["batch_size"],
            input_img_mode=config["input_img_mode"],
            input_key=config["input_key"],
            target_key=config["target_key"],
        )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = config["mixup"] > 0 or config["cutmix"] > 0.0 or config["cutmix_minmax"] is not None
    if mixup_active:
        mixup_args = {
            "mixup_alpha": config["mixup"],
            "cutmix_alpha": config["cutmix"],
            "cutmix_minmax": config["cutmix_minmax"],
            "prob": config["mixup_prob"],
            "switch_prob": config["mixup_switch_prob"],
            "mode": config["mixup_mode"],
            "label_smoothing": config["smoothing"],
            "num_classes": config["num_classes"],
        }
        if prefetcher:
            assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = config["train_interpolation"]
    if config["no_aug"] or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=config["batch_size"],
        is_training=True,
        no_aug=config["no_aug"],
        re_prob=config["reprob"],
        re_mode=config["remode"],
        re_count=config["recount"],
        re_split=config["resplit"],
        train_crop_mode=config["train_crop_mode"],
        scale=config["scale"],
        ratio=config["ratio"],
        hflip=config["hflip"],
        vflip=config["vflip"],
        color_jitter=config["color_jitter"],
        color_jitter_prob=config["color_jitter_prob"],
        grayscale_prob=config["grayscale_prob"],
        gaussian_blur_prob=config["gaussian_blur_prob"],
        auto_augment=config["aa"],
        num_aug_repeats=config["aug_repeats"],
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=config["workers"],
        distributed=distributed,
        collate_fn=collate_fn,
        pin_memory=config["pin_mem"],
        device=torch.device(config["device"]),
        use_prefetcher=prefetcher,
        use_multi_epochs_loader=config["use_multi_epochs_loader"],
        worker_seeding=config["worker_seeding"],
    )

    loader_eval = None
    if config["val_split"]:
        eval_workers = config["workers"]
        if distributed:
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, config["workers"])
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=config["validation_batch_size"] or config["batch_size"],
            is_training=False,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=eval_workers,
            distributed=distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=config["pin_mem"],
            device=torch.device(config["device"]),
            use_prefetcher=prefetcher,
        )

    # setup loss function
    if config["jsd_loss"]:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=config["smoothing"])
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if config["bce_loss"]:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=config["bce_target_thresh"],
                sum_classes=config["bce_sum"],
                pos_weight=config["bce_pos_weight"],
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif config["smoothing"]:
        if config["bce_loss"]:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=config["smoothing"],
                target_threshold=config["bce_target_thresh"],
                sum_classes=config["bce_sum"],
                pos_weight=config["bce_pos_weight"],
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=config["smoothing"])
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=config["device"])
    validate_loss_fn = nn.CrossEntropyLoss().to(device=config["device"])

    # setup checkpoint saver and eval metric tracking
    eval_metric = config["eval_metric"] if loader_eval is not None else "loss"
    decreasing_metric = eval_metric == "loss"
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if rank == 0:
        if config["experiment"]:
            exp_name = config["experiment"]
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(config["model"]),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(config["output"] if config["output"] else "./output/train", exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=SimpleNamespace(model=model),
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=config["checkpoint_hist"],
        )
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            f.write(json.dumps(config, indent=2))

    if rank == 0 and config["log_wandb"]:
        if config["has_wandb"]:
            wandb.init(project=config["experiment"], config=config)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + config["grad_accum_steps"] - 1) // config["grad_accum_steps"]
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(SimpleNamespace(**config), decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if rank == 0:
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
        )

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch_cls(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                prefetcher,
                distributed,
                world_size,
                rank,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            if distributed and dist_bn in ("broadcast", "reduce"):
                if rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, world_size, dist_bn == "reduce")

            if loader_eval is not None:
                eval_metrics = validate_cls(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    prefetcher,
                    distributed,
                    world_size,
                    rank,
                    amp_autocast=amp_autocast,
                )

                if model_ema is not None and not model_ema_force_cpu:
                    if distributed and dist_bn in ("broadcast", "reduce"):
                        utils.distribute_bn(model_ema, world_size, dist_bn == "reduce")

                    ema_eval_metrics = validate_cls(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        prefetcher,
                        distributed,
                        world_size,
                        rank,
                        amp_autocast=amp_autocast,
                        log_suffix=" (EMA)",
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, "summary.csv"),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": eval_metrics,
                }
            )

    except KeyboardInterrupt:
        pass

    results = {"all": results}
    if best_metric is not None:
        results["best"] = results["all"][best_epoch - start_epoch]
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
    print(f"--result\n{json.dumps(results, indent=4)}")

    login(token=os.getenv("HF_TOKEN"), new_session=False)
    push_to_hf_hub(
        model,
        model_hub_name,
        model_config={"label_names": CLASSES},
    )

    if distributed:
        destroy_process_group()


def train_one_epoch_cls(  # noqa: C901
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    prefetcher,
    distributed,
    world_size,
    rank,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
    num_updates_total=None,
):
    if config["mixup_off_epoch"] and epoch >= config["mixup_off_epoch"]:
        if prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = config["grad_accum_steps"]
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not prefetcher:
            input, target = input.to(config["device"]), target.to(config["device"])
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if config["channels_last"]:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)  # noqa: B023
                loss = loss_fn(output, target % config["num_classes"])  # noqa: B023
            if accum_steps > 1:  # noqa: B023
                loss /= accum_steps  # noqa: B023
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=config["clip_grad"],
                    clip_mode=config["clip_mode"],
                    parameters=model_parameters(model, exclude_head="agc" in config["clip_mode"]),
                    create_graph=second_order,
                    need_update=need_update,  # noqa: B023
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:  # noqa: B023
                    if config["clip_grad"] is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head="agc" in config["clip_mode"]),
                            value=config["clip_grad"],
                            mode=config["clip_mode"],
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if config["synchronize_step"] and torch.device(config["device"]).type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % config["log_interval"] == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if distributed:
                reduced_loss = utils.reduce_tensor(loss.data, world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                update_sample_count *= world_size

            if rank == 0:
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

                if config["save_images"] and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx), padding=0, normalize=True
                    )

        if saver is not None and config["recovery_interval"] and ((update_idx + 1) % config["recovery_interval"] == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate_cls(
    model,
    loader,
    loss_fn,
    prefetcher,
    distributed,
    world_size,
    rank,
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not prefetcher:
                input = input.to(config["device"])
                target = target.to(config["device"])
            if config["channels_last"]:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = config["tta"]
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0 : target.size(0) : reduce_factor]

                loss = loss_fn(output, target % config["num_classes"])
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if distributed:
                reduced_loss = utils.reduce_tensor(loss.data, world_size)
                acc1 = utils.reduce_tensor(acc1, world_size)
                acc5 = utils.reduce_tensor(acc5, world_size)
            else:
                reduced_loss = loss.data

            if torch.device(config["device"]).type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if rank == 0 and (last_batch or batch_idx % config["log_interval"] == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                )

    metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])

    return metrics


def push_to_hub(local_dir: str, model_name: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(local_dir)
    model.push_to_hub(HF_USERNAME + "/" + model_name)
    processor.push_to_hub(HF_USERNAME + "/" + model_name)


# -----------------------------------------------------------------------------

# main


def main(cls: bool, sft: bool, dpo: bool):
    if not cls and not sft and not dpo:
        raise ValueError("Must specify at least one of `cls`, `sft`, or `dpo`")

    if cls:
        run_cls()
    if sft:
        os.chdir(TRAIN_REPO_PATH)
        _exec_subprocess(
            [
                "cp",
                f"{DATA_VOL_PATH}/{SFT_DATA}",
                f"data/{SFT_DATA}",
            ]
        )
        _exec_subprocess(
            [
                "llamafactory-cli",
                "train",
                SFT_YAML,
            ]
        )
        push_to_hub(f"{RUNS_VOL_PATH}/{SFT_MODEL}", SFT_MODEL)
        _exec_subprocess(
            [
                "llamafactory-cli",
                "export",
                SFT_MERGE_YAML,
            ]
        )
        push_to_hub(f"{RUNS_VOL_PATH}/{SFT_MODEL}-merged", f"{SFT_MODEL}-merged")
        # checkpoint_folder = str(
        #     max(
        #         list((Path(f"{RUNS_VOL_PATH}/{SFT_MODEL}).glob("checkpoint-*"))),
        #         key=lambda x: int(x.name.split("-")[-1]),
        #     )
        # )
        # push_to_hub(checkpoint_folder, SFT_MODEL)
    if dpo:
        os.chdir(TRAIN_REPO_PATH)
        _exec_subprocess(
            [
                "cp",
                f"{DATA_VOL_PATH}/{DPO_DATA}",
                f"data/{DPO_DATA}",
            ]
        )
        _exec_subprocess(
            [
                "llamafactory-cli",
                "train",
                DPO_YAML,
            ]
        )
        push_to_hub(f"{RUNS_VOL_PATH}/{DPO_MODEL}", DPO_MODEL)
        _exec_subprocess(
            [
                "llamafactory-cli",
                "export",
                DPO_MERGE_YAML,
            ]
        )
        push_to_hub(f"{RUNS_VOL_PATH}/{DPO_MODEL}-merged", f"{DPO_MODEL}-merged")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=TRAIN_TIMEOUT,
)
def run(cls: bool, sft: bool, dpo: bool):
    main(cls, sft, dpo)


@app.local_entrypoint()
def local(cls: bool = False, sft: bool = False, dpo: bool = False):
    run.remote(cls, sft, dpo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    args = parser.parse_args()
    main(args.cls, args.sft, args.dpo)


# TODO:
# - recreate in pytorch and inc torchtitan
# - replace slow act + fns with custom CUDA kernels
#   - better yet: recreate in C/Cuda
