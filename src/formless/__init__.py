import traceback
import os
import tempfile
import time
from uuid import uuid4
from term_image.image import from_file
import requests
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated


DEFAULT_IMG_URL = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
DEFAULT_QUESTION = "What is the content of this image?"
API_URL = "https://andrewhinh--formless-api-modal-get-infer.modal.run"

# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False}


# Fns
def run() -> None:
    image_url, question = state["image_url"], state["question"]
    response = requests.post(f"{API_URL}/api-key")
    assert response.ok, response.status_code
    api_key = response.json()
    response = requests.post(
        API_URL, json={"image_url": image_url, "question": question}, headers={"X-API-Key": api_key}
    )
    assert response.ok, response.status_code
    return response.json()


@app.command(
    help="Handwritten + image OCR.",
    epilog="Made by [bold blue]Andrew Hinh.[/bold blue] :mechanical_arm::person_climbing:",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": True},
)
def main(
    image_url: Annotated[
        str, typer.Option("--image-url", "-i", help="Image URL", rich_help_panel="Inputs")
    ] = DEFAULT_IMG_URL,
    question: Annotated[
        str, typer.Option("--question", "-q", help="Question", rich_help_panel="Inputs")
    ] = DEFAULT_QUESTION,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="Verbose mode", rich_help_panel="General")
    ] = 0,
):
    try:
        start = time.monotonic_ns()
        request_id = uuid4()

        state.update(
            {
                "image_url": image_url,
                "question": question,
                "verbose": verbose > 0,
            }
        )

        if state["verbose"]:
            response = requests.get(image_url)
            image_filename = image_url.split("/")[-1]
            image_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}-{image_filename}")
            with open(image_path, "wb") as file:
                file.write(response.content)
            terminal_image = from_file(image_path)
            terminal_image.draw()
            print(f"[bold blue]{question}[/bold blue]")

        if state["verbose"]:
            print("[red]Press[/red] [blue]Ctrl+C[/blue] [red]to stop at any time.[/red]")
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
            ) as progress:
                progress.add_task(f"Generating response to request {request_id}", total=None)
                generated_text = run()
        else:
            generated_text = run()
        print(f"[bold green]{generated_text}[/bold green]")

        if state["verbose"]:
            print(
                f"[red]request[/red] [blue]{request_id}[/blue] [red]completed in[/red] [blue]{round((time.monotonic_ns() - start) / 1e9, 2)}[/blue] [red]seconds[/red]"
            )

    except KeyboardInterrupt:
        if state["verbose"]:
            print("[red]\n\nExiting...[/red]")
    except Exception as e:
        if state["verbose"]:
            print(f"[red]Failed with error: {e}[/red]")
            print(traceback.format_exc())
            print("[red]\n\nExiting...[/red]")


# TODO:
# - add multiple uploads/urls
