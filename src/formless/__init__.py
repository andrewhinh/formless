from secrets import token_urlsafe
from notifypy import Notify
import time
import base64
import io
import json

from pathlib import Path

import torch
from PIL import Image
import mss
import traceback

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated

from .utils import (
    setup_classifier,
    class_config,
    setup_gguf,
)

# -----------------------------------------------------------------------------

# Classifier
CLASSIFIER = "hf_hub:andrewhinh/resnet152-224-Screenspot"

# LLMs
SYSTEM_PROMPT = "You are a helpful assistant."

## MM-LLM
MM_LLM = "abetlen/nanollava-gguf"
MM_LLM_CLIP = "nanollava-mmproj-f16.gguf"
MM_LLM_GGUF = "nanollava-text-model-f16.gguf"

### https://github.com/abetlen/llama-cpp-python/discussions/319
MM_LLM_CTX = 1024  # img + text tokens
MM_LLM_TEMPERATURE = 0.2
MM_LLM_TOP_P = 0.95
MM_LLM_MIN_P = 0.05
MM_LLM_TYPICAL_P = 1.0
MM_LLM_TOP_K = 40
MM_LLM_MAX_TOKENS = 32768
MM_LLM_PRESENCE_PENALTY = 0
MM_LLM_FREQUENCY_PENALTY = 0
MM_LLM_REPEAT_PENALTY = 1.1
MM_LLM_TFS_Z = 1

MM_LLM_PROMPT = "Describe the image."
MM_LLM_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"screen_text": {"type": "string"}},
        "required": ["screen_text"],
    },
}

## LLM
LLM = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
LLM_GGUF = "llama-3.2-1b-instruct-q4_k_m.gguf"

LLM_CTX = 1024  # text tokens
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.95
LLM_MIN_P = 0.05
LLM_TYPICAL_P = 1.0
LLM_TOP_K = 40
LLM_MAX_TOKENS = 32768
LLM_PRESENCE_PENALTY = 0
LLM_FREQUENCY_PENALTY = 0
LLM_REPEAT_PENALTY = 1.1
LLM_TFS_Z = 1

LLM_TITLE_PROMPT = "Here's a description of the user's screen: {description}. Write a short (2-5 words) title about refocusing on work, noting the description."
LLM_MESSAGE_PROMPT = "Here's a description of the user's screen: {description}. Write a short (max 15 words) message about refocusing, noting the description."
LLM_TITLE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
}
LLM_MESSAGE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}

# -----------------------------------------------------------------------------

# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False}

# -----------------------------------------------------------------------------

# Notifypy
NOTIFICATION_INTERVAL = 8  # seconds
notification = Notify(
    default_application_name="Modeldemo",
    default_notification_urgency="critical",
    default_notification_icon=str(Path(__file__).parent / "icon.png"),
    default_notification_audio=str(Path(__file__).parent / "sound.wav"),
)

# -----------------------------------------------------------------------------


# Helper fns
def capture_screenshot() -> Image:
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def image_to_base64_data_uri(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


def classify(model, transforms, amp_autocast, image: Image):
    t0 = time.time()

    device = torch.device(class_config["device"])
    img_pt = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with amp_autocast():
            output = model(img_pt)
    output = output.softmax(-1)
    output, indices = output.topk(class_config["topk"])
    labels = model.pretrained_cfg["label_names"]
    predictions = [{"label": labels[i], "score": v.item()} for i, v in zip(indices, output, strict=False)]
    preds, probs = [p["label"] for p in predictions], [p["score"] for p in predictions]

    t1 = time.time()
    if state["verbose"]:
        print(f"Prediction: {preds[0]}")
        print(f"Probability: ({probs[0] * 100:.2f}%) in {t1 - t0:.2f} seconds")
    return preds[0]


def generate(
    llm,
    system,
    temperature,
    top_p,
    min_p,
    typical_p,
    top_k,
    max_tokens,
    presence_penalty,
    frequency_penalty,
    repeat_penalty,
    tfs_z,
    prompt,
    response_format,
    image=None,
) -> str:
    t0 = time.time()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    if image:
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image)}})

    generated_text = llm.create_chat_completion(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        typical_p=typical_p,
        top_k=top_k,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
    )["choices"][0]["message"]["content"]
    if state["verbose"]:
        print(f"Generated text: {generated_text}")

    t1 = time.time()
    if state["verbose"]:
        print()
        print(f"Tok/sec: {len(generated_text) / (t1 - t0):.2f}")

    return generated_text


# -----------------------------------------------------------------------------


# Typer CLI
def run() -> None:
    ## load models
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading models...", total=None)
            cls_tsfm, amp_autocast, classifier = setup_classifier(state["classifier"], state["verbose"])
            ocr = setup_gguf(
                state["mm_llm"], state["mm_llm_gguf"], state["mm_llm_ctx"], state["mm_llm_clip"], state["verbose"]
            )
            llm = setup_gguf(state["llm"], state["llm_gguf"], state["llm_ctx"], verbose=state["verbose"])
    else:
        cls_tsfm, amp_autocast, classifier = setup_classifier(state["classifier"], state["verbose"])
        ocr = setup_gguf(
            state["mm_llm"], state["mm_llm_gguf"], state["mm_llm_ctx"], state["mm_llm_clip"], state["verbose"]
        )
        llm = setup_gguf(state["llm"], state["llm_gguf"], state["llm_ctx"], verbose=state["verbose"])
    classifier.eval()

    ## main loop
    while True:
        img = capture_screenshot()
        pred = classify(classifier, cls_tsfm, amp_autocast, img)

        if pred == "distracted":
            ocr_out = generate(
                llm=ocr,
                system=state["system_prompt"],
                temperature=state["mm_llm_temperature"],
                top_p=state["mm_llm_top_p"],
                min_p=state["mm_llm_min_p"],
                typical_p=state["mm_llm_typical_p"],
                top_k=state["mm_llm_top_k"],
                max_tokens=state["mm_llm_max_tokens"],
                presence_penalty=state["mm_llm_presence_penalty"],
                frequency_penalty=state["mm_llm_frequency_penalty"],
                repeat_penalty=state["mm_llm_repeat_penalty"],
                tfs_z=state["mm_llm_tfs_z"],
                prompt=state["mm_llm_prompt"],
                response_format=MM_LLM_FORMAT,
                image=img,
            )
            try:
                ocr_out = json.loads(ocr_out)
                screen_text = ocr_out["screen_text"]
            except KeyError:
                if state["verbose"]:
                    print("No screen text detected.")
                continue

            title_out = generate(
                llm=llm,
                system=state["system_prompt"],
                temperature=state["llm_temperature"],
                top_p=state["llm_top_p"],
                min_p=state["llm_min_p"],
                typical_p=state["llm_typical_p"],
                top_k=state["llm_top_k"],
                max_tokens=state["llm_max_tokens"],
                presence_penalty=state["llm_presence_penalty"],
                frequency_penalty=state["llm_frequency_penalty"],
                repeat_penalty=state["llm_repeat_penalty"],
                tfs_z=state["llm_tfs_z"],
                prompt=state["llm_title_prompt"].format(description=screen_text),
                response_format=LLM_TITLE_FORMAT,
            )
            message_out = generate(
                llm=llm,
                system=state["system_prompt"],
                temperature=state["llm_temperature"],
                top_p=state["llm_top_p"],
                min_p=state["llm_min_p"],
                typical_p=state["llm_typical_p"],
                top_k=state["llm_top_k"],
                max_tokens=state["llm_max_tokens"],
                presence_penalty=state["llm_presence_penalty"],
                frequency_penalty=state["llm_frequency_penalty"],
                repeat_penalty=state["llm_repeat_penalty"],
                tfs_z=state["llm_tfs_z"],
                prompt=state["llm_message_prompt"].format(description=screen_text),
                response_format=LLM_MESSAGE_FORMAT,
            )
            try:
                title_out = json.loads(title_out)
                message_out = json.loads(message_out)
                title = title_out["title"]
                message = message_out["message"]
            except KeyError:
                if state["verbose"]:
                    print("No title or message generated.")
                continue

            notification.title = title
            notification.message = message
            notification.send(block=False)
            time.sleep(NOTIFICATION_INTERVAL)


@app.command(
    help="Stay [bold red]focused.[/bold red]",
    epilog="Made by [bold blue]Andrew Hinh.[/bold blue] :mechanical_arm::person_climbing:",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": True},
)
def main(
    system_prompt: Annotated[
        str, typer.Option("--system-prompt", "-s", help="System prompt for LLM", rich_help_panel="System")
    ] = SYSTEM_PROMPT,
    classifier: Annotated[
        str,
        typer.Option(
            "--classifier",
            "-c",
            help="Focused/distracted classifier, timm-compatible model on HF Hub",
            rich_help_panel="Classifier",
        ),
    ] = CLASSIFIER,
    mm_llm: Annotated[
        str,
        typer.Option(
            "--mm-llm",
            "-m",
            help="MM-LLM for OCR, llama-cpp-python compatible multimodal model on HF Hub",
            rich_help_panel="MM_LLM",
        ),
    ] = MM_LLM,
    mm_llm_clip: Annotated[
        str, typer.Option("--mm-llm-clip", "-mc", help="CLIP path in HF repo", rich_help_panel="MM_LLM")
    ] = MM_LLM_CLIP,
    mm_llm_gguf: Annotated[
        str, typer.Option("--mm-llm-gguf", "-mg", help="LLM GGUF path in HF repo", rich_help_panel="MM_LLM")
    ] = MM_LLM_GGUF,
    mm_llm_ctx: Annotated[
        int, typer.Option("--mm-llm-ctx", "-mx", help="MM-LLM context size", rich_help_panel="MM_LLM")
    ] = MM_LLM_CTX,
    mm_llm_temperature: Annotated[
        float, typer.Option("--mm-llm-temperature", "-mt", help="Temperature for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_TEMPERATURE,
    mm_llm_top_p: Annotated[
        float, typer.Option("--mm-llm-top-p", "-mtp", help="Top P for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_TOP_P,
    mm_llm_min_p: Annotated[
        float, typer.Option("--mm-llm-min-p", "-mmp", help="Min P for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_MIN_P,
    mm_llm_typical_p: Annotated[
        float, typer.Option("--mm-llm-typ-p", "-mtypp", help="Typical P for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_TYPICAL_P,
    mm_llm_top_k: Annotated[
        int, typer.Option("--mm-llm-top-k", "-mtk", help="Top K for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_TOP_K,
    mm_llm_max_tokens: Annotated[
        int, typer.Option("--mm-llm-max-tokens", "-mmt", help="Max tokens for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_MAX_TOKENS,
    mm_llm_presence_penalty: Annotated[
        float,
        typer.Option("--mm-llm-presence-penalty", "-mpp", help="Presence penalty for MM-LLM", rich_help_panel="MM_LLM"),
    ] = MM_LLM_PRESENCE_PENALTY,
    mm_llm_frequency_penalty: Annotated[
        float,
        typer.Option(
            "--mm-llm-frequency-penalty", "-mfp", help="Frequency penalty for MM-LLM", rich_help_panel="MM_LLM"
        ),
    ] = MM_LLM_FREQUENCY_PENALTY,
    mm_llm_repeat_penalty: Annotated[
        float,
        typer.Option("--mm-llm-repeat-penalty", "-mrp", help="Repeat penalty for MM-LLM", rich_help_panel="MM_LLM"),
    ] = MM_LLM_REPEAT_PENALTY,
    mm_llm_tfs_z: Annotated[
        float, typer.Option("--mm-llm-tfs-z", "-mtz", help="TFS Z for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_TFS_Z,
    mm_llm_prompt: Annotated[
        str, typer.Option("--mm-llm-prompt", "-mmp", help="Prompt for MM-LLM", rich_help_panel="MM_LLM")
    ] = MM_LLM_PROMPT,
    llm: Annotated[
        str,
        typer.Option(
            "--llm", "-l", help="LLM for reply, llama-cpp-python compatible model on HF Hub", rich_help_panel="LLM"
        ),
    ] = LLM,
    llm_gguf: Annotated[
        str, typer.Option("--llm-gguf", "-lg", help="LLM GGUF path in HF repo", rich_help_panel="LLM")
    ] = LLM_GGUF,
    llm_ctx: Annotated[int, typer.Option("--llm-ctx", "-lx", help="LLM context size", rich_help_panel="LLM")] = LLM_CTX,
    llm_temperature: Annotated[
        float, typer.Option("--llm-temperature", "-lt", help="Temperature for LLM", rich_help_panel="LLM")
    ] = LLM_TEMPERATURE,
    llm_top_p: Annotated[
        float, typer.Option("--llm-top-p", "-ltp", help="Top P for LLM", rich_help_panel="LLM")
    ] = LLM_TOP_P,
    llm_min_p: Annotated[
        float, typer.Option("--llm-min-p", "-lmp", help="Min P for LLM", rich_help_panel="LLM")
    ] = LLM_MIN_P,
    llm_typical_p: Annotated[
        float, typer.Option("--llm-typ-p", "-ltyp", help="Typical P for LLM", rich_help_panel="LLM")
    ] = LLM_TYPICAL_P,
    llm_top_k: Annotated[
        int, typer.Option("--llm-top-k", "-ltk", help="Top K for LLM", rich_help_panel="LLM")
    ] = LLM_TOP_K,
    llm_max_tokens: Annotated[
        int, typer.Option("--llm-max-tokens", "-lmt", help="Max tokens for LLM", rich_help_panel="LLM")
    ] = LLM_MAX_TOKENS,
    llm_presence_penalty: Annotated[
        float, typer.Option("--llm-presence-penalty", "-lpp", help="Presence penalty for LLM", rich_help_panel="LLM")
    ] = LLM_PRESENCE_PENALTY,
    llm_frequency_penalty: Annotated[
        float, typer.Option("--llm-frequency-penalty", "-lfp", help="Frequency penalty for LLM", rich_help_panel="LLM")
    ] = LLM_FREQUENCY_PENALTY,
    llm_repeat_penalty: Annotated[
        float, typer.Option("--llm-repeat-penalty", "-lrp", help="Repeat penalty for LLM", rich_help_panel="LLM")
    ] = LLM_REPEAT_PENALTY,
    llm_tfs_z: Annotated[
        float, typer.Option("--llm-tfs-z", "-ltz", help="TFS Z for LLM", rich_help_panel="LLM")
    ] = LLM_TFS_Z,
    llm_title_prompt: Annotated[
        str, typer.Option("--llm-title-prompt", "-ltp", help="Prompt for title generation", rich_help_panel="LLM")
    ] = LLM_TITLE_PROMPT,
    llm_message_prompt: Annotated[
        str, typer.Option("--llm-message-prompt", "-lmp", help="Prompt for message generation", rich_help_panel="LLM")
    ] = LLM_MESSAGE_PROMPT,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="Verbose mode", rich_help_panel="General")
    ] = 0,
):
    try:
        state.update(
            {
                "system_prompt": system_prompt,
                "classifier": classifier,
                "mm_llm": mm_llm,
                "mm_llm_clip": mm_llm_clip,
                "mm_llm_gguf": mm_llm_gguf,
                "mm_llm_ctx": mm_llm_ctx,
                "mm_llm_temperature": mm_llm_temperature,
                "mm_llm_top_p": mm_llm_top_p,
                "mm_llm_min_p": mm_llm_min_p,
                "mm_llm_typical_p": mm_llm_typical_p,
                "mm_llm_top_k": mm_llm_top_k,
                "mm_llm_max_tokens": mm_llm_max_tokens,
                "mm_llm_presence_penalty": mm_llm_presence_penalty,
                "mm_llm_frequency_penalty": mm_llm_frequency_penalty,
                "mm_llm_repeat_penalty": mm_llm_repeat_penalty,
                "mm_llm_tfs_z": mm_llm_tfs_z,
                "mm_llm_prompt": mm_llm_prompt,
                "llm": llm,
                "llm_gguf": llm_gguf,
                "llm_ctx": llm_ctx,
                "llm_temperature": llm_temperature,
                "llm_top_p": llm_top_p,
                "llm_min_p": llm_min_p,
                "llm_typical_p": llm_typical_p,
                "llm_top_k": llm_top_k,
                "llm_max_tokens": llm_max_tokens,
                "llm_presence_penalty": llm_presence_penalty,
                "llm_frequency_penalty": llm_frequency_penalty,
                "llm_repeat_penalty": llm_repeat_penalty,
                "llm_tfs_z": llm_tfs_z,
                "llm_title_prompt": llm_title_prompt,
                "llm_message_prompt": llm_message_prompt,
                "verbose": verbose > 0,
            }
        )
        run()
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print(traceback.format_exc())
            print("\n\nExiting...")
