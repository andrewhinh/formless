# https://github.com/huggingface/pytorch-image-models/blob/main/inference.py

import logging
import os
import time
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ft.utils import ARTIFACT_PATH, CLASSES
from timm.data import ImageNetInfo, create_dataset, create_loader, infer_imagenet_subset, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, set_jit_fuser, setup_default_logging

try:
    from apex import amp  # noqa: F401

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
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError:
    has_functorch = False

has_compile = hasattr(torch, "compile")


_FMT_EXT = {
    "json": ".json",
    "json-record": ".json",
    "json-split": ".json",
    "parquet": ".parquet",
    "csv": ".csv",
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("inference")

# -----------------------------------------------------------------------------

# Dataset parameters
data_dir = ARTIFACT_PATH / "data"  # path to dataset (root dir)
dataset = ""  # dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)
split = "validation"  # dataset split (default: validation)
workers = os.cpu_count() // 2 + 1  # number of data loading workers (default: 2)

# Model parameters
model = "resnet152"  # Name of model to train (default: "resnet50")
pretrained = False  # use pre-trained model
checkpoint = (
    ARTIFACT_PATH / "runs" / "20240922-103145-resnet152-224" / "model_best.pth.tar"
)  # path to latest checkpoint (default: none)
num_classes = len(CLASSES)  # Number classes in dataset
img_size = None  # Input image dimension, uses model default if empty
in_chans = None  # Number of input image channels (default: None => model default)
input_size = None  # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
use_train_size = False  # force use of train input size, even when test size is specified in pretrained cfg
crop_pct = None  # Input image center crop pct
mean = None  # Override mean pixel value of dataset
std = None  # Override std deviation of dataset
interpolation = ""  # Image resize interpolation type (overrides model)
batch_size = 256  # mini-batch size (default: 256)
channels_last = False  # Use channels_last memory layout
fuser = ""  # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')
model_kwargs = {}  # Additional model keyword arguments

# scripting / codegen
torchscript = False  # torch.jit.script the full model
torchcompile = None  # Enable compilation w/ specified backend (default: inductor).
aot_autograd = False  # Enable AOT Autograd support.

# Device & distributed
device = "cpu"  # Device (accelerator) to use.
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
num_gpu = torch.cuda.device_count() if device == "cuda" else 0  # Number of GPUS to use
amp = False  # use Native AMP for mixed precision training
amp_dtype = "float16"  # lower precision AMP dtype (default: float16)

# Misc
log_freq = 1  # batch logging frequency (default: 10)
test_pool = False  # enable test time pool
results_dir = ARTIFACT_PATH / "runs"  # folder for output results
results_file = None  # results filename (relative to results-dir)
results_format = ["csv"]  # results format (one of "csv", "json", "json-split", "parquet")
results_separate_col = False  # separate output columns per result index.
topk = 1  # Top-k to output to CSV
fullname = False  # use full sample name in output (not just basename).
filename_col = "filename"  # name for filename / sample name column
index_col = "index"  # name for output indices column(s)
label_col = "label"  # name for output indices column(s)
output_col = None  # name for logit/probs output column(s)
output_type = "prob"  # output type column ("prob" for probabilities, "logit" for raw logits)
label_type = "description"  # type of label to output, one of  "none", "name", "description", "detailed"
include_index = False  # include the class index in results
exclude_output = False  # exclude logits/probs from results, just indices. topk must be set !=0.

# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, Path, type(None)))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}  # since Path not serializable

# -----------------------------------------------------------------------------


def main():  # noqa: C901
    setup_default_logging()

    config["pretrained"] = config["pretrained"] or not config["checkpoint"]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(config["device"])

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if config["amp"]:
        assert has_native_amp, "Please update PyTorch to a version with native AMP (or use APEX)."
        assert config["amp_dtype"] in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info("Running inference in mixed precision with native PyTorch AMP.")
    else:
        _logger.info("Running inference in float32. AMP not enabled.")

    if config["fuser"]:
        set_jit_fuser(config["fuser"])

    # create model
    in_chans = 3
    if config["in_chans"] is not None:
        in_chans = config["in_chans"]
    elif config["input_size"] is not None:
        in_chans = config["input_size"][0]

    model = create_model(
        config["model"],
        num_classes=config["num_classes"],
        in_chans=in_chans,
        pretrained=config["pretrained"],
        checkpoint_path=config["checkpoint"],
        **config["model_kwargs"],
    )
    if config["num_classes"] is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        config["num_classes"] = model.num_classes

    _logger.info(f"Model {config['model']} created, param count: {sum([m.numel() for m in model.parameters()])}")

    data_config = resolve_data_config(config, model=model)
    test_time_pool = False
    if config["test_pool"]:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()
    if config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    if config["torchscript"]:
        model = torch.jit.script(model)
    elif config["torchcompile"]:
        assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        torch._dynamo.reset()
        model = torch.compile(model, backend=config["torchcompile"])
    elif config["aot_autograd"]:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if config["num_gpu"] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(config["num_gpu"])))

    root_dir = config["data_dir"]
    dataset = create_dataset(
        root=root_dir,
        name=config["dataset"],
        split=config["split"],
    )

    if test_time_pool:
        data_config["crop_pct"] = 1.0

    workers = 1 if "tfds" in config["dataset"] or "wds" in config["dataset"] else config["workers"]
    loader = create_loader(
        dataset,
        batch_size=config["batch_size"],
        use_prefetcher=True,
        num_workers=workers,
        device=device,
        **data_config,
    )

    to_label = None
    if config["label_type"] in ("name", "description", "detail"):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if config["label_type"] == "name":

                def to_label(x):
                    return dataset_info.index_to_label_name(x)
            elif config["label_type"] == "detail":

                def to_label(x):
                    return dataset_info.index_to_description(x, detailed=True)
            else:

                def to_label(x):
                    return dataset_info.index_to_description(x)

            to_label = np.vectorize(to_label)
        else:
            _logger.error("Cannot deduce ImageNet subset from model, no labelling will be performed.")

    top_k = min(config["topk"], config["num_classes"])
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    use_probs = config["output_type"] == "prob"
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            with amp_autocast():
                output = model(input)

            if use_probs:
                output = output.softmax(-1)

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if config["include_index"]:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % config["log_freq"] == 0:
                _logger.info(
                    "Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        batch_idx, len(loader), batch_time=batch_time
                    )
                )

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not config["fullname"])

    output_col = config["output_col"] or ("prob" if use_probs else "logit")
    data_dict = {config["filename_col"]: filenames}
    if config["results_separate_col"] and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f"{config['index_col']}_{i}"] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f"{config['label_col']}_{i}"] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f"{output_col}_{i}"] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[config["index_col"]] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[config["label_col"]] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    results_filename = config["results_file"]
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f"{config['model']}-{img_size}"

    if config["results_dir"]:
        results_filename = os.path.join(config["results_dir"], results_filename)

    for fmt in config["results_format"]:
        save_results(df, results_filename, fmt)

    print("--result")
    print(df.set_index(config["filename_col"]).to_json(orient="index", indent=4))


def save_results(df, results_filename, results_format="csv", filename_col="filename"):
    results_filename += _FMT_EXT[results_format]
    if results_format == "parquet":
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == "json":
        df.set_index(filename_col).to_json(results_filename, indent=4, orient="index")
    elif results_format == "json-records":
        df.to_json(results_filename, lines=True, orient="records")
    elif results_format == "json-split":
        df.to_json(results_filename, indent=4, orient="split", index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    main()
