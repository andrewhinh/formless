from pathlib import Path

import modal

from utils import (
    API_URL,
    DATA_VOLUME,
    MINUTES,
    NAME,
    PYTHON_VERSION,
)

parent_path: Path = Path(__file__).parent

# Modal
FE_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(  # add Python dependencies
        "python-fasthtml==0.6.10", "simpleicons==7.21.0", "requests==2.32.3", "sqlite-utils==3.18"
    )
    .copy_local_dir(parent_path, "/root/")
)

FE_TIMEOUT = 24 * 60 * MINUTES  # max
FE_CONTAINER_IDLE_TIMEOUT = 20 * MINUTES  # max
FE_ALLOW_CONCURRENT_INPUTS = 1000  # max


APP_NAME = f"{NAME}-frontend"
app = modal.App(APP_NAME)


@app.function(
    image=FE_IMAGE,
    secrets=[modal.Secret.from_dotenv(path=parent_path)],
    timeout=FE_TIMEOUT,
    container_idle_timeout=FE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=FE_ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():  # noqa: C901
    import os
    import uuid

    import requests
    from fasthtml import common as fh
    from simpleicons.icons import si_github, si_pypi

    fasthtml_app, rt = fh.fast_app(
        ws_hdr=True,
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
    )
    fh.setup_toasts(fasthtml_app)

    # gens database for storing generated image details
    os.makedirs(f"/{DATA_VOLUME}/frontend", exist_ok=True)
    tables = fh.database(f"/{DATA_VOLUME}/frontend/gens.db").t
    gens = tables.gens
    if gens not in tables:
        gens.create(image_url=str, response=str, session_id=str, id=int, folder=str, pk="id")
    Generation = gens.dataclass()

    # Show the image (if available) for a generation
    def generation_preview(g, session):
        if "session_id" not in session:
            return "No session ID"
        if g.session_id != session["session_id"]:
            return "Wrong session ID!"
        grid_cls = "box col-xs-12 col-sm-6 col-md-4 col-lg-3"
        if g.response:
            return fh.Div(
                fh.Card(
                    fh.Img(src=g.image_url, alt="Card image", cls="card-img-top"),
                    fh.Div(fh.P(fh.B("Response: "), g.response, cls="card-text"), cls="card-body"),
                ),
                id=f"gen-{g.id}",
                cls=grid_cls,
            )
        return fh.Div(
            f"Scanning image '{g.image_url}'...",
            id=f"gen-{g.id}",
            hx_get=f"/gens/{g.id}",
            hx_trigger="every 2s",
            hx_swap="outerHTML",
            cls=grid_cls,
        )

    # Components
    def icon(
        svg,
        width="35",
        height="35",
        viewBox="0 0 15 15",
        fill="none",
        cls="rounded p-0.5 hover:bg-zinc-700 cursor-pointer",
    ):
        return fh.Svg(
            fh.NotStr(svg),
            width=width,
            height=height,
            viewBox=viewBox,
            fill=fill,
            cls=cls,
        )

    # Layout
    def nav():
        return fh.Nav(
            fh.A(
                f"{NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Div(
                fh.A(
                    icon(si_github.svg),
                    href="https://github.com/andrewhinh/formless",
                    target="_blank",
                ),
                fh.A(
                    icon(si_pypi.svg),
                    href="https://pypi.org/project/formless/",
                    target="_blank",
                ),
                cls="flex gap-4",
            ),
            cls="flex justify-between p-4",
        )

    def main_content(session):
        gen_containers = [
            generation_preview(g, session) for g in gens(limit=10, where=f"session_id == '{session['session_id']}'")
        ]
        return fh.Main(
            fh.Form(
                fh.Group(
                    fh.Input(
                        id="new-img-url",
                        name="img_url",
                        placeholder="Enter an image url",
                        cls="text-blue-300 hover:text-blue-100 p-4 placeholder:text-blue-300",
                    ),
                    fh.Button("Scan"),
                    cls="max-w-md self-center",
                ),
                hx_post="/",
                target_id="gen-list",
                hx_swap="afterbegin",
            ),
            fh.Div(*gen_containers[::-1], id="gen-list", cls="flex"),
            cls="flex flex-col gap-4 p-4",
        )

    def footer():
        return fh.Footer(
            fh.P("Made by", cls="text-white text-lg"),
            fh.A(
                "Andrew Hinh",
                href="https://andrewhinh.github.io/",
                cls="text-blue-300 text-lg font-bold hover:text-blue-100",
            ),
            cls="justify-end text-right p-4",
        )

    # Routes
    @rt("/")
    async def home(session):
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        return fh.Title(NAME), fh.Div(
            nav(),
            main_content(session),
            footer(),
            cls="flex flex-col justify-between min-h-screen bg-zinc-900 w-full",
        )

    # A pending preview keeps polling this route until we return the image preview
    @rt("/gens/{id}")
    def preview(id: int, session):
        return generation_preview(gens.get(id), session)

    # For images, CSS, etc.
    @rt("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    # Generation route
    @fasthtml_app.post("/")
    def post(image_url: str, session):
        # Check for session ID
        if "session_id" not in session:
            return "No session ID"

        clear_input = fh.Input(
            id="new-image-url", name="image-url", placeholder="Enter an image url", hx_swap_oob="true"
        )

        # Generate as before
        g = gens.insert(Generation(image_url=image_url, session_id=session["session_id"]))
        generate_and_save(g)

        return generation_preview(g, session), clear_input

    # Generate the response (in a separate thread)
    @fh.threaded
    def generate_and_save(g) -> None:
        response = requests.post(API_URL, json={"image_url": g.image_url})
        assert response.ok, response.status_code
        g.response = response.json()

    return fasthtml_app
