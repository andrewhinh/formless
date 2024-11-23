import os
from datetime import datetime
from pathlib import Path

import modal

from utils import (
    DATA_VOLUME,
    MINUTES,
    NAME,
    PYTHON_VERSION,
    VOLUME_CONFIG,
)

parent_path: Path = Path(__file__).parent
in_prod = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"

# Modal
FE_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(  # add Python dependencies
        "python-fasthtml==0.6.10",
        "simpleicons==7.21.0",
        "requests==2.32.3",
        "sqlite-utils==3.18",
        "stripe==11.1.0",
        "validators==0.34.0",
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
    secrets=[modal.Secret.from_dotenv(path=parent_path, filename=".env" if in_prod else ".env.dev")],
    timeout=FE_TIMEOUT,
    container_idle_timeout=FE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=FE_ALLOW_CONCURRENT_INPUTS,
    volumes=VOLUME_CONFIG,
)
@modal.asgi_app()
def modal_get():  # noqa: C901
    import csv
    import io
    import uuid

    import requests
    import stripe
    import validators
    from fasthtml import common as fh
    from simpleicons.icons import si_github, si_pypi

    # setup
    f_app, _ = fh.fast_app(
        ws_hdr=True,
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
        boost=True,
    )
    fh.setup_toasts(f_app)

    ## db
    upload_dir = Path(f"/{DATA_VOLUME}/uploads")
    upload_dir.mkdir(exist_ok=True)
    db_path = f"/{DATA_VOLUME}/main.db"
    # TODO: uncomment for debugging
    # os.remove(db_path)
    tables = fh.database(db_path).t

    ### generations
    gens = tables.gens
    if gens not in tables:
        gens.create(
            request_at=str,
            image_url=str,
            image_file=str,
            question=str,
            failed=bool,
            response=str,
            session_id=str,
            id=int,
            pk="id",
        )
    Generation = gens.dataclass()

    def get_curr_gens(session):
        curr_gens = gens(where=f"session_id == '{session['session_id']}'")
        curr_gens = [g for g in curr_gens if not g.failed]  # TODO: limitation of sqlite-utils
        return curr_gens

    ### api keys
    api_keys = tables.api_keys
    if api_keys not in tables:
        api_keys.create(key=str, granted_at=str, session_id=str, id=int, pk="id")
    ApiKey = api_keys.dataclass()

    ### global balance
    init_balance = 100
    global_balance = tables.global_balance
    if global_balance not in tables:
        global_balance.create(balance=int, pk="id")
    Balance = global_balance.dataclass()

    def get_curr_balance():
        try:
            return global_balance.get(1)
        except Exception:
            global_balance.insert(Balance(balance=init_balance))
            return global_balance.get(1)

    ## stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    DOMAIN: str = os.environ["DOMAIN"]

    # ui

    ## components
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

    ## preview while waiting for response
    def generation_preview(g, session):
        ### check if g and session are valid
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        try:
            gens.get(g.id)
        except Exception:
            return None
        if g.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        image_src = None
        if g.image_url:
            image_src = g.image_url
        elif g.image_file:
            temp_path = parent_path / Path(g.image_file).name
            with open(temp_path, "wb") as f:
                f.write(open(g.image_file, "rb").read())
            image_src = f"/{Path(g.image_file).name}"

        if g.failed:
            return None
        elif g.response:
            return (
                fh.Card(
                    fh.Img(
                        src=image_src,
                        alt="Card image",
                        cls="w-20 object-contain",
                    ),
                    fh.Div(
                        fh.P(
                            g.question,
                            cls="text-blue-300",
                        ),
                        fh.P(
                            g.response,
                            onclick="navigator.clipboard.writeText(this.innerText);",
                            hx_post="/toast?message=Copied to clipboard!&type=success",
                            hx_target="#toast-container",
                            hx_swap="outerHTML",
                            cls="text-green-300 hover:text-green-100 cursor-pointer max-w-full",
                            title="Click to copy",
                        ),
                        cls="flex flex-col gap-2",
                    ),
                    cls="w-full flex gap-4",
                    style="max-height: 40vh; overflow-y: auto;",
                    id=f"gen-{g.id}",
                ),
            )
        return fh.Card(
            fh.Img(
                src=image_src,
                alt="Card image",
                cls="w-20 object-contain",
            ),
            fh.Div(
                fh.P(
                    g.question,
                    cls="text-blue-300",
                ),
                fh.P("Scanning image ..."),
                cls="flex flex-col gap-2",
            ),
            cls="w-full flex gap-4",
            style="max-height: 40vh;",
            id=f"gen-{g.id}",
            hx_get=f"/gens/{g.id}",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
        )

    def key_request_preview(k, session):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        try:
            api_keys.get(k.id)
        except Exception:
            return None
        if k.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        if k.key and k.granted_at:
            obscured_key = k.key[:4] + "*" * (len(k.key) - 4)
            short_key = obscured_key[:8] + "..."

            return (
                fh.Div(
                    fh.Div(
                        obscured_key,
                        onmouseover=(
                            f"if (window.innerWidth >= 768) {{"
                            f" this.innerText = '{k.key}'; "
                            f"}} else {{"
                            f" this.innerText = '{short_key}'; "
                            f"}}"
                        ),
                        onmouseout=(
                            f"if (window.innerWidth >= 768) {{"
                            f" this.innerText = '{obscured_key}'; "
                            f"}} else {{"
                            f" this.innerText = '{short_key}'; "
                            f"}}"
                        ),
                        onclick=f"navigator.clipboard.writeText('{k.key}');",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_target="#toast-container",
                        hx_swap="outerHTML",
                        cls="text-blue-300 hover:text-blue-100 cursor-pointer w-2/3",
                        title="Click to copy",
                        id=f"key-element-{k.id}",
                    ),
                    fh.Div(
                        k.granted_at,
                        cls="w-1/3",
                    ),
                    id=f"key-{k.id}",
                    cls="flex p-2",
                ),
                fh.Script(
                    f"""
                    function updateKeyDisplay() {{
                        var element = document.getElementById('key-element-{k.id}');
                        if (element) {{
                            if (window.innerWidth >= 768) {{
                                element.innerText = '{obscured_key}';
                            }} else {{
                                element.innerText = '{short_key}';
                            }}
                        }}
                    }}

                    window.onresize = updateKeyDisplay;
                    window.onload = updateKeyDisplay;
                    updateKeyDisplay();
                    """
                ),
            )
        return fh.Div(
            fh.Div("Requesting key ...", cls="w-2/3"),
            fh.Div("", cls="w-1/3"),
            id=f"key-{k.key}",
            cls="flex p-2",
            hx_get=f"/keys/{k.id}",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
        )

    ## layout
    def nav():
        return fh.Nav(
            fh.A(
                f"{NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Div(
                fh.A(
                    "Developer",
                    href="/developer",
                    cls="text-lg text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
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
                    cls="flex flex-row gap-4",
                ),
                cls="flex flex-col items-end md:flex-row md:items-center gap-2 md:gap-8",
            ),
            cls="flex justify-between p-4",
            style="max-height: 10vh;",
        )

    def main_content(session):
        curr_gens = get_curr_gens(session)
        gen_containers = [generation_preview(g, session) for g in curr_gens]
        return fh.Main(
            fh.Div(
                fh.Div(
                    get_gen_form_url_button(session),
                    get_gen_form_upload_button(session),
                    cls="w-full flex flex-col md:flex-row gap-2 md:gap-4",
                ),
                get_gen_form("image-url", session),
                cls="w-2/3 flex flex-col gap-4 justify-center items-center",
            ),
            fh.Div(
                get_clear_gens_button(session),  # clear all button, hidden if no gens
                get_export_gens_button(session),  # export to csv button, hidden if no gens
                cls="flex flex-col md:flex-row justify-center gap-2 md:gap-4 w-2/3",
            ),
            fh.Div(
                *gen_containers[::-1],
                id="gen-list",
                cls="flex flex-col justify-center items-center gap-2 w-2/3",
                style="max-height: 40vh; overflow-y: auto;",
            ),
            cls="flex flex-col justify-center items-center gap-4 p-8",
            style="max-height: 80vh;",
        )

    def developer_page(session):
        key_containers = [
            key_request_preview(key, session)
            for key in api_keys(limit=10, where=f"session_id == '{session['session_id']}'")
        ]
        return fh.Main(
            fh.Button(
                "Request New Key",
                id="request-new-key",
                hx_post="/request-key",
                target_id="api-key-table",
                hx_swap="afterbegin",
                cls="text-blue-300 hover:text-blue-100 p-2 w-2/3 border-blue-300 border-2 hover:border-blue-100",
            ),
            fh.Div(
                get_clear_keys_button(session),
                get_export_keys_button(session),
                cls="flex flex-col md:flex-row justify-center gap-4 w-2/3",
            ),
            fh.Div(
                fh.Div(
                    fh.Div("Key", cls="font-bold w-2/3"),
                    fh.Div("Granted At", cls="font-bold w-1/3"),
                    cls="flex p-2",
                ),
                fh.Div(
                    *key_containers[::-1],
                    id="api-key-table",
                    style="max-height: 40vh; overflow-y: auto;",
                ),
                cls="w-2/3 flex flex-col gap-2 text-sm md:text-lg border-slate-500 border-2",
            ),
            cls="flex flex-col justify-center items-center gap-4 p-8",
            style="max-height: 80vh;",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            get_balance(),  # live-updating balance
            fh.Div(
                fh.P("Made by"),
                fh.A(
                    "Andrew Hinh",
                    href="https://andrewhinh.github.io/",
                    cls="font-bold text-blue-300 hover:text-blue-100",
                ),
                cls="flex flex-col text-right gap-0.5",
            ),
            cls="flex justify-between p-4 text-sm md:text-lg",
            style="max-height: 10vh;",
        )

    # background tasks (separate threads)
    ## generation
    @fh.threaded
    def generate_and_save(session, g):
        k = api_keys.insert(ApiKey(key=None, granted_at=None, session_id=g.session_id))
        generate_key_and_save(k)

        if g.image_url:
            response = requests.post(
                os.getenv("API_URL"),
                json={"image_url": g.image_url, "question": g.question},
                headers={"X-API-Key": k.key},
            )
        elif g.image_file:
            response = requests.post(
                f"{os.getenv('API_URL')}/upload",
                data=open(g.image_file, "rb").read(),
                headers={
                    "X-API-Key": k.key,
                    "Content-Type": "application/octet-stream",
                    "question": g.question,
                },
            )

        # TODO: uncomment for debugging
        # g.response = "temp"
        # gens.update(g)
        # return

        # TODO: uncomment for debugging
        # response = requests.Response()
        # response.status_code = 500

        if not response.ok:
            fh.add_toast(session, "Failed with status code: " + str(response.status_code), "error")
            g.failed = True
        else:
            g.response = response.json()
        gens.update(g)

    ## key generation
    @fh.threaded
    def generate_key_and_save(k) -> None:
        k.key = str(uuid.uuid4())
        k.granted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        api_keys.update(k)

    # routes
    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## toasts without target
    @f_app.post("/toast")
    async def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return fh.Div(id="toast-container", cls="hidden")

    ## home page
    @f_app.get("/")
    async def home(session):
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        if "gen_form" not in session:
            session["gen_form"] = "image-url"
        return (
            fh.Title(NAME),
            fh.Div(
                nav(),
                main_content(session),
                toast_container(),
                footer(),
                cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
            ),
            fh.Script(
                """
                document.addEventListener('htmx:beforeRequest', (event) => {
                    if (event.target.id === 'export-gens-csv') {
                        event.preventDefault();
                        window.location.href = "/export-gens";
                    }
                });
            """
            ),
        )

    ## developer page
    @f_app.get("/developer")
    def developer(session):
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        return (
            fh.Title(NAME + " | " + "developer"),
            fh.Div(
                nav(),
                developer_page(session),
                toast_container(),
                footer(),
                cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
            ),
            fh.Script(
                """
                document.addEventListener('htmx:beforeRequest', (event) => {
                    if (event.target.id === 'export-keys-csv') {
                        event.preventDefault();
                        window.location.href = "/export-keys";
                    }
                });
            """
            ),
        )

    ## pending previews keeps polling routes until response is ready
    @f_app.get("/get-gen-form-url-button")
    def get_gen_form_url_button(session):
        curr_gen_form = session["gen_form"]
        return fh.Div(
            fh.Button(
                "Image URL",
                id="get-gen-form-url",
                cls="w-full h-full text-blue-100 bg-blue-500 p-2 border-blue-500 border-2"
                if curr_gen_form == "image-url"
                else "w-full h-full text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                hx_post="/get-gen-form/image-url",
                hx_target="#gen-form",
                hx_swap="innerHTML",
            ),
            id="get-gen-form-url-container",
            hx_get="/get-gen-form-url-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.get("/get-gen-form-upload-button")
    def get_gen_form_upload_button(session):
        curr_gen_form = session["gen_form"]
        return fh.Div(
            fh.Button(
                "Image Upload",
                id="get-gen-form-upload",
                cls="w-full h-full text-blue-100 bg-blue-500 p-2 border-blue-500 border-2"
                if curr_gen_form == "image-upload"
                else "w-full h-full text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                hx_post="/get-gen-form/image-upload",
                hx_target="#gen-form",
                hx_swap="innerHTML",
            ),
            id="get-gen-form-upload-container",
            hx_get="/get-gen-form-upload-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.post("/get-gen-form/{view}")
    def get_gen_form(view: str, session):
        session["gen_form"] = view
        return (
            fh.Form(
                fh.Div(
                    fh.Input(
                        id="new-image-url",
                        name="image_url",  # passed to fn call for python syntax
                        placeholder="Enter an image url",
                    ),
                    fh.Input(
                        id="new-question",
                        name="question",
                        placeholder="Specify format or question",
                    ),
                    fh.Button(
                        "Scan",
                        type="submit",
                        cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                    ),
                    cls="flex flex-col md:gap-2",
                ),
                hx_post="/url",
                target_id="gen-list",
                hx_swap="afterbegin",
                id="gen-form",
                cls="w-full h-full",
            )
            if view == "image-url"
            else fh.Form(
                fh.Div(
                    fh.Input(
                        id="new-image-upload",
                        name="image_file",
                        type="file",
                        accept="image/*",
                    ),
                    fh.Input(
                        id="new-question",
                        name="question",
                        placeholder="Specify format or question",
                    ),
                    fh.Button(
                        "Scan",
                        type="submit",
                        cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                    ),
                    cls="flex flex-col gap-4",
                ),
                hx_post="/upload",
                target_id="gen-list",
                hx_swap="afterbegin",
                id="gen-form",
                cls="w-full h-full",
            ),
        )

    @f_app.get("/clear-gens-button")
    def get_clear_gens_button(session):
        curr_gens = get_curr_gens(session)
        return fh.Div(
            fh.Button(
                "Clear all",
                id="clear-gens",
                hx_delete="/gens",
                hx_target="body",
                hx_push_url="true",
                hx_confirm="Are you sure?",
                cls="text-red-300 hover:text-red-100 p-2 border-red-300 border-2 hover:border-red-100 w-full h-full",
            )
            if curr_gens
            else None,
            id="clear-gens-button-container",
            hx_get="/clear-gens-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.get("/export-gens-button")
    def get_export_gens_button(session):
        curr_gens = get_curr_gens(session)
        return fh.Div(
            fh.Button(
                "Export to CSV",
                id="export-gens-csv",
                hx_get="/export-gens",
                hx_target="this",
                hx_swap="none",
                hx_boost="false",
                cls="text-green-300 hover:text-green-100 p-2 border-green-300 border-2 hover:border-green-100 w-full h-full",
            )
            if curr_gens
            else None,
            id="export-gens-button-container",
            hx_get="/export-gens-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.get("/gens/{id}")
    def preview(id: int, session):
        return generation_preview(gens.get(id), session)

    @f_app.get("/clear-keys-button")
    def get_clear_keys_button(session):
        curr_keys = api_keys(where=f"session_id == '{session['session_id']}'")
        return fh.Div(
            fh.Button(
                "Clear all",
                id="clear-keys",
                hx_delete="/keys",
                hx_target="body",
                hx_push_url="true",
                hx_confirm="Are you sure?",
                cls="text-red-300 hover:text-red-100 p-2 border-red-300 border-2 hover:border-red-100 w-full h-full",
            )
            if curr_keys
            else None,
            id="clear-keys-button-container",
            hx_get="/clear-keys-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.get("/export-keys-button")
    def get_export_keys_button(session):
        curr_keys = api_keys(where=f"session_id == '{session['session_id']}'")
        return fh.Div(
            fh.Button(
                "Export to CSV",
                id="export-keys-csv",
                hx_get="/export-keys",
                hx_target="this",
                hx_swap="none",
                hx_boost="false",
                cls="text-green-300 hover:text-green-100 p-2 border-green-300 border-2 hover:border-green-100 w-full h-full",
            )
            if curr_keys
            else None,
            id="export-keys-button-container",
            hx_get="/export-keys-button",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex items-center justify-center w-full h-full",
        )

    @f_app.get("/keys/{id}")
    def key_request(id: int, session):
        return key_request_preview(api_keys.get(id), session)

    @f_app.get("/balance")
    def get_balance():
        curr_balance = get_curr_balance()
        return fh.Div(
            fh.Div(
                fh.P("Global balance:"),
                fh.P(f"{curr_balance.balance} credits", cls="font-bold"),
                cls="flex items-start gap-0.5 md:gap-1",
            ),
            fh.P(
                fh.A("Buy 50 more", href="/buy_global", cls="font-bold text-blue-300 hover:text-blue-100"),
                " to share ($1)",
            ),
            id="balance",
            hx_get="/balance",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="flex flex-col gap-0.5",
        )

    ## generation route
    @f_app.post("/url")
    def generate_from_url(session, image_url: str, question: str):
        # validation
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if not image_url:
            fh.add_toast(session, "No image URL provided", "error")
            return None
        if not validators.url(image_url):
            fh.add_toast(session, "Invalid image URL", "error")
            return None

        # Warn if we're out of balance
        curr_balance = get_curr_balance()
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.balance -= 1
        global_balance.update(curr_balance)

        # Clear input
        clear_img_input = fh.Input(
            id="new-image-url", name="image_url", placeholder="Enter an image url", hx_swap_oob="true"
        )
        clear_q_input = fh.Input(
            id="new-question", name="question", placeholder="Specify format or question", hx_swap_oob="true"
        )

        # Generate as before
        g = gens.insert(
            Generation(
                request_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_url=image_url,
                question=question,
                session_id=session["session_id"],
            )
        )
        generate_and_save(session, g)
        return generation_preview(g, session), clear_img_input, clear_q_input

    @f_app.post("/upload")
    async def generate_from_upload(session, image_file: fh.UploadFile, question: str):
        # Check for session ID
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        # Ensure extension is valid image
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        file_extension = Path(image_file.filename).suffix.lower()
        if file_extension not in valid_extensions:
            fh.add_toast(session, "Invalid file type. Please upload an image.", "error")
            return None

        # Limit img size
        max_size_mb = 5
        max_size_bytes = max_size_mb * 1024 * 1024
        if len(await image_file.read()) > max_size_bytes:
            fh.add_toast(session, f"File size exceeds {max_size_mb}MB limit.", "error")
            return None

        # Write file to disk
        filebuffer = await image_file.read()
        upload_path = upload_dir / str(uuid.uuid4())
        upload_path.write_bytes(filebuffer)

        # Warn if we're out of balance
        curr_balance = get_curr_balance()
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.balance -= 1
        global_balance.update(curr_balance)

        # Clear input
        clear_img_input = fh.Input(
            id="new-image-upload", name="image_file", type="file", accept="image/*", hx_swap_oob="true"
        )
        clear_q_input = fh.Input(
            id="new-question", name="question", placeholder="Specify format or question", hx_swap_oob="true"
        )

        # Generate as before
        g = gens.insert(
            Generation(
                request_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_file=str(upload_path),
                question=question,
                session_id=session["session_id"],
            )
        )
        generate_and_save(session, g)

        return generation_preview(g, session), clear_img_input, clear_q_input

    ## api key request
    @f_app.post("/request-key")
    def generate_key(session):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        k = api_keys.insert(ApiKey(key=None, granted_at=None, session_id=session["session_id"]))
        generate_key_and_save(k)
        return key_request_preview(k, session)

    ## clear gens
    @f_app.delete("/gens")
    def clear_all(session):
        ids = [g.id for g in gens(where=f"session_id == '{session['session_id']}'")]
        for id in ids:
            gens.delete(id)
        fh.add_toast(session, "Deleted generations.", "success")
        return fh.RedirectResponse("/", status_code=303)

    ## clear keys
    @f_app.delete("/keys")
    def clear_keys(session):
        ids = [k.id for k in api_keys(where=f"session_id == '{session['session_id']}'")]
        for id in ids:
            api_keys.delete(id)
        fh.add_toast(session, "Deleted keys.", "success")
        return fh.RedirectResponse("/developer", status_code=303)

    ## export gens to CSV
    @f_app.get("/export-gens")
    async def export_gens(req):
        session = req.session
        curr_gens = get_curr_gens(session)
        if not curr_gens:
            return fh.Response("No generations found.", media_type="text/plain")

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["request_at", "image_url", "image_file", "question", "response", "failed"])
        for g in curr_gens:
            writer.writerow([g.request_at, g.image_url, Path(g.image_file).name, g.question, g.response, g.failed])

        output.seek(0)
        response = fh.Response(
            output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=gens.csv"}
        )
        return response

    ## export keys to CSV
    @f_app.get("/export-keys")
    async def export_keys(req):
        session = req.session
        curr_keys = api_keys(where=f"session_id == '{session['session_id']}'")
        if not curr_keys:
            return fh.Response("No keys found.", media_type="text/plain")

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["key", "granted_at"])
        for k in curr_keys:
            writer.writerow([k.key, k.granted_at])

        output.seek(0)
        response = fh.Response(
            output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=keys.csv"}
        )
        return response

    ## stripe
    ### send the user here to buy credits
    @f_app.get("/buy_global")
    def buy_credits(session):
        if "session_id" not in session:
            return "Error no session id"

        s = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": 100,
                        "product_data": {
                            "name": "Buy 50 credits for $1 (to share)",
                        },
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=DOMAIN + "/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=DOMAIN + "/cancel",
        )
        ### send the USER to STRIPE
        return fh.RedirectResponse(s["url"])

    ### STRIPE sends the USER here after a payment was canceled.
    @f_app.get("/cancel")
    def cancel():
        return fh.RedirectResponse("/")

    ### STRIPE sends the USER here after a payment was 'successful'.
    @f_app.get("/success")
    def success():
        return fh.RedirectResponse("/")

    ### STRIPE calls this to tell APP when a payment was completed.
    @f_app.post("/webhook")
    async def stripe_webhook(request):
        # print(request)
        # print("Received webhook")
        payload = await request.body()
        payload = payload.decode("utf-8")
        signature = request.headers.get("stripe-signature")
        # print(payload)

        # verify the Stripe webhook signature
        try:
            event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
        except ValueError:
            # print("Invalid payload")
            return {"error": "Invalid payload"}, 400
        except stripe.error.SignatureVerificationError:
            # print("Invalid signature")
            return {"error": "Invalid signature"}, 400

        # handle the event
        if event["type"] == "checkout.session.completed":
            # session = event["data"]["object"]
            # print("Session completed", session)
            curr_balance = get_curr_balance()
            curr_balance.balance += 50
            global_balance.update(curr_balance)
            return {"status": "success"}, 200

    return f_app


# TODO:
# - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#   - Only allow authorized users to upload files:
#       - add user authentication: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
#       - add form validation: https://hypermedia.systems/htmx-patterns/#_next_steps_validating_contact_emails
#   - Run the file through an antivirus or a sandbox if available to validate that it doesn't contain malicious data
#   - Run the file through CDR (Content Disarm & Reconstruct) if applicable type (PDF, DOCX, etc...)
#   - Protect the file upload from CSRF attacks: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
# - replace polling routes with SSE + oob: https://docs.fastht.ml/tutorials/quickstart_for_web_devs.html#server-sent-events-sse
# - add smooth db migrations: prob switch to sqlmodel + alembic

# - add multiple file urls/uploads: https://docs.fastht.ml/tutorials/quickstart_for_web_devs.html#multiple-file-uploads
# - add better infinite scroll: https://hypermedia.systems/htmx-patterns/#_another_application_improvement_paging
# - add gens/keys counts: https://hypermedia.systems/more-htmx-patterns/#_lazy_loading
# - add granular delete: https://hypermedia.systems/more-htmx-patterns/#_inline_delete
# - add bulk delete: https://hypermedia.systems/more-htmx-patterns/#_bulk_delete
# - add animations: https://hypermedia.systems/a-dynamic-archive-ui/#_smoothing_things_out_animations_in_htmx
