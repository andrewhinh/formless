from pathlib import Path

import modal

from utils import (
    API_URL,
    DATA_VOLUME,
    MINUTES,
    NAME,
    PYTHON_VERSION,
    VOLUME_CONFIG,
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
    volumes=VOLUME_CONFIG,
)
@modal.asgi_app()
def modal_get():  # noqa: C901
    import os
    import uuid

    import requests
    from fasthtml import common as fh
    from simpleicons.icons import si_github, si_pypi

    # FastHTML setup
    f_app, _ = fh.fast_app(
        ws_hdr=True,
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
    )
    fh.setup_toasts(f_app)

    # database
    in_prod = not os.getenv("LIVE", False) and not os.getenv("DEBUG", False)
    db_path = f"/{DATA_VOLUME}/frontend/gens.db" if in_prod else f"/{DATA_VOLUME}/frontend/gens_dev.db"
    # TODO: uncomment for debugging
    # os.remove(db_path)
    os.makedirs(f"/{DATA_VOLUME}/frontend", exist_ok=True)
    tables = fh.database(db_path).t
    gens = tables.gens
    if gens not in tables:
        gens.create(image_url=str, response=str, session_id=str, id=int, folder=str, pk="id")
    Generation = gens.dataclass()

    # preview while waiting for response
    def generation_preview(g, session):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if g.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if g.response:
            fh.add_toast(session, "Scan complete", "success")
            return (
                fh.Card(
                    fh.Img(src=g.image_url, alt="Card image", cls="w-80 object-contain"),
                    fh.P(
                        g.response,
                        onclick="navigator.clipboard.writeText(this.innerText);",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_target="#toast-container",
                        hx_swap="outerHTML",
                        cls="text-blue-300 hover:text-blue-100 cursor-pointer",
                        title="Click to copy",
                    ),
                    cls="w-2/3 flex flex-col justify-center items-center gap-4",
                    id=f"gen-{g.id}",
                ),
            )
        return fh.Card(
            fh.Img(src=g.image_url, alt="Card image", cls="w-80 object-contain"),
            fh.P("Scanning image ..."),
            cls="w-2/3 flex flex-col justify-center items-center gap-4",
            id=f"gen-{g.id}",
            hx_get=f"/gens/{g.id}",
            hx_trigger="every 2s",
            hx_swap="outerHTML",
        )

    # components
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

    # layout
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
                        id="new-image-url",
                        name="image_url",
                        placeholder="Enter an image url",
                        cls="p-2",
                    ),
                    fh.Button("Scan", cls="text-blue-300 hover:text-blue-100 p-2"),
                ),
                hx_post="/",
                target_id="gen-list",
                hx_swap="afterbegin",
                cls="w-2/3",
            ),
            fh.Div(*gen_containers[::-1], id="gen-list", cls="flex flex-col justify-center items-center gap-4"),
            cls="flex flex-col justify-center items-center gap-4",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            fh.P("Made by", cls="text-lg"),
            fh.A(
                "Andrew Hinh",
                href="https://andrewhinh.github.io/",
                cls="text-blue-300 text-lg font-bold hover:text-blue-100",
            ),
            cls="justify-end text-right p-4",
        )

    # routes
    @f_app.get("/")
    async def home(session):
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        return fh.Title(NAME), fh.Div(
            nav(),
            main_content(session),
            toast_container(),
            footer(),
            cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
        )

    ## pending preview keeps polling this route until we return the image preview
    @f_app.get("/gens/{id}")
    def preview(id: int, session):
        return generation_preview(gens.get(id), session)

    ## toasts without target
    @f_app.post("/toast")
    async def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return fh.Div(id="toast-container", cls="hidden")

    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## generation route
    @f_app.post("/")
    def generate(image_url: str, session):
        # Check for session ID
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        # Check image URL
        if not image_url:
            fh.add_toast(session, "No image URL provided", "error")
            return None
        if requests.head(image_url).headers["content-type"] not in ("image/png", "image/jpeg", "image/jpg"):
            fh.add_toast(session, "Invalid image URL", "error")
            return None

        clear_input = fh.Input(
            id="new-image-url", name="image-url", placeholder="Enter an image url", hx_swap_oob="true"
        )

        # Generate as before
        g = gens.insert(Generation(image_url=image_url, session_id=session["session_id"]))
        generate_and_save(g)

        return generation_preview(g, session), clear_input

    ## generate the response (in a separate thread)
    @fh.threaded
    def generate_and_save(g) -> None:
        response = requests.post(API_URL, json={"image_url": g.image_url})
        assert response.ok, response.status_code
        g.response = response.json()
        gens.update(g)

    return f_app
