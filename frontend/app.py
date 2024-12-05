import os
import secrets
from pathlib import Path

import modal

from utils import (
    DATA_VOLUME,
    FAVICON_PATH,  # noqa: F401, so available to browser
    IN_PROD,
    MINUTES,
    NAME,
    PYTHON_VERSION,
    VOLUME_CONFIG,
)

parent_path: Path = Path(__file__).parent


# Modal
def run_alembic():  # since volumes are needed to access db
    import subprocess

    subprocess.run(  # noqa: S602
        [  # noqa: S607
            # "if [ -d /{DATA_VOLUME}/versions ]; then cp -r /{DATA_VOLUME}/versions/* /root/db/migrations/versions/; else mkdir -p /root/db/migrations/versions; fi",  # TODO: figure out why versioning leads to empty db
            f"export PYTHONPATH=/root/ && alembic -c /root/db/migrations/alembic.ini revision --autogenerate -m {NAME}",
            "export PYTHONPATH=/root/ && alembic -c /root/db/migrations/alembic.ini upgrade head",
            # f"cp -r /root/db/migrations/versions/* /{DATA_VOLUME}/versions/",
        ],
        shell=True,
    )


SECRETS = [modal.Secret.from_dotenv(path=parent_path, filename=".env" if IN_PROD else ".env.dev")]
IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .run_commands(["git clone https://github.com/Len-Stevens/Python-Antivirus.git"])
    .pip_install(  # add Python dependencies
        "python-fasthtml==0.6.10",
        "simpleicons==7.21.0",
        "requests==2.32.3",
        "stripe==11.1.0",
        "validators==0.34.0",
        "pillow==11.0.0",
        "alembic==1.14.0",
        "sqlmodel==0.0.22",
        "modal==0.67.25",
    )
    .copy_local_dir(parent_path.parent / "db", "/root/db")
    .copy_local_file(parent_path.parent / "utils.py", "/root/utils.py")
    .run_function(run_alembic, secrets=SECRETS, volumes=VOLUME_CONFIG, force_build=True)
)

FE_TIMEOUT = 24 * 60 * MINUTES  # max
FE_CONTAINER_IDLE_TIMEOUT = 20 * MINUTES  # max
FE_ALLOW_CONCURRENT_INPUTS = 1000  # max


APP_NAME = f"{NAME}-frontend"
app = modal.App(APP_NAME)


@app.function(
    image=IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=FE_TIMEOUT,
    container_idle_timeout=FE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=FE_ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():  # noqa: C901
    import csv
    import io
    import subprocess
    import uuid
    from asyncio import sleep

    import requests
    import stripe
    import validators
    from fasthtml import common as fh
    from PIL import Image
    from simpleicons.icons import si_github, si_pypi
    from sqlmodel import Session as DBSession
    from sqlmodel import select
    from starlette.middleware.cors import CORSMiddleware

    from db.models import (
        ApiKey,
        ApiKeyCreate,
        ApiKeyRead,
        Gen,
        GenCreate,
        GenRead,
        GlobalBalance,
        GlobalBalanceCreate,
        GlobalBalanceRead,
        get_db_session,
        init_balance,
    )

    # setup
    f_app, _ = fh.fast_app(
        ws_hdr=True,
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
            fh.Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
        boost=True,
    )
    fh.setup_toasts(f_app)
    f_app.add_middleware(
        CORSMiddleware,
        allow_origins=["/"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## db
    upload_dir = Path(f"/{DATA_VOLUME}/uploads")
    upload_dir.mkdir(exist_ok=True)
    os.chmod(upload_dir, 0o600)  # Read/write by owner only

    def get_curr_gens(session: dict, db_session: DBSession) -> list[Gen]:
        return db_session.exec(
            select(Gen).where(Gen.session_id == session["session_id"]).order_by(Gen.request_at)
        ).all()

    def get_curr_keys(session: dict, db_session: DBSession) -> list[ApiKey]:
        return db_session.exec(select(ApiKey).where(ApiKey.session_id == session["session_id"])).all()

    def get_curr_balance(db_session: DBSession) -> GlobalBalance:
        try:
            return db_session.exec(select(GlobalBalance)).first()
        except Exception:
            new_balance = GlobalBalanceCreate(balance=init_balance)
            db_session.add(new_balance)
            db_session.commit()
            db_session.refresh(new_balance)
            return new_balance

    ## stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    DOMAIN: str = os.environ["DOMAIN"]

    ## SSE state
    shutdown_event = fh.signal_shutdown()
    shown_generations = {}
    global shown_balance
    shown_balance = 0

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

    def gen_view(g: GenRead, session: dict, db_session: DBSession):
        ### check if g and session are valid
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        present = db_session.exec(select(Gen).where(Gen.id == g.id)).first()
        if not present:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if g.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        image_src = None
        if g.image_url and validate_image_url(g.image_url):
            image_src = g.image_url
        elif g.image_file and isinstance(validate_image_file(image_file=None, upload_path=Path(g.image_file)), Path):
            temp_path = parent_path / Path(g.image_file).name
            with open(temp_path, "wb") as f:
                f.write(open(g.image_file, "rb").read())
            image_src = f"/{Path(g.image_file).name}"

        if g.failed:
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
                    fh.P(
                        "Generation failed",
                        cls="text-red-300 ",
                    ),
                    cls="flex flex-col gap-2",
                ),
                cls="w-full flex gap-4",
                style="max-height: 40vh;",
                id=f"gen-{g.id}",
            )
        elif g.response:
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
                    fh.P(
                        g.response,
                        onclick="navigator.clipboard.writeText(this.innerText);",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_indicator="#spinner",
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
        )

    def key_view(k: ApiKeyRead, session: dict, db_session: DBSession):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        present = db_session.exec(select(ApiKey).where(ApiKey.id == k.id)).first()
        if not present:
            fh.add_toast(session, "Please refresh the page", "error")
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
                        hx_indicator="#spinner",
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
        )

    def balance_view(gb: GlobalBalanceRead, session: dict, db_session: DBSession) -> tuple:
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        present = db_session.exec(select(GlobalBalance).where(GlobalBalance.id == gb.id)).first()
        if not present:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        return (
            fh.P("Global balance:"),
            fh.P(f"{gb.balance} credits", cls="font-bold"),
        )

    def gen_form_toggle(gen_form: str, hx_swap_oob: bool = "false") -> fh.Div:
        return fh.Div(
            fh.Button(
                "Image URL",
                id="gen-form-toggle-url",
                cls="w-full h-full text-blue-100 bg-blue-500 p-2 border-blue-500 border-2"
                if gen_form == "image-url"
                else "w-full h-full text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                hx_get="/get-gen-form/image-url",
                hx_indicator="#spinner",
                hx_target="#gen-form",
                hx_swap="innerHTML",
            ),
            fh.Button(
                "Image Upload",
                id="gen-form-toggle-upload",
                cls="w-full h-full text-blue-100 bg-blue-500 p-2 border-blue-500 border-2"
                if gen_form == "image-upload"
                else "w-full h-full text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                hx_get="/get-gen-form/image-upload",
                hx_indicator="#spinner",
                hx_target="#gen-form",
                hx_swap="innerHTML",
            ),
            id="gen-form-toggle",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-full flex flex-col md:flex-row gap-2 md:gap-4",
        )

    def gen_manage(curr_gens: list[GenRead], hx_swap_oob: bool = "false") -> fh.Div:
        return fh.Div(
            fh.Button(
                "Clear all",
                hx_delete="/gens",
                hx_indicator="#spinner",
                hx_target="body",
                hx_push_url="true",
                hx_confirm="Are you sure?",
                cls="text-red-300 hover:text-red-100 p-2 border-red-300 border-2 hover:border-red-100 w-full h-full",
            )
            if curr_gens
            else None,
            fh.Button(
                "Export to CSV",
                id="export-gens-csv",
                hx_get="/export-gens",
                hx_indicator="#spinner",
                hx_target="this",
                hx_swap="none",
                hx_boost="false",
                cls="text-green-300 hover:text-green-100 p-2 border-green-300 border-2 hover:border-green-100 w-full h-full",
            )
            if curr_gens
            else None,
            id="gen-manage",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex flex-col md:flex-row justify-center gap-2 md:gap-4 w-2/3",
        )

    def key_manage(curr_keys: list[ApiKeyRead], hx_swap_oob: bool = "false") -> fh.Div:
        return fh.Div(
            fh.Button(
                "Clear all",
                hx_delete="/keys",
                hx_indicator="#spinner",
                hx_target="body",
                hx_push_url="true",
                hx_confirm="Are you sure?",
                cls="text-red-300 hover:text-red-100 p-2 border-red-300 border-2 hover:border-red-100 w-full h-full",
            )
            if curr_keys
            else None,
            fh.Button(
                "Export to CSV",
                id="export-keys-csv",
                hx_get="/export-keys",
                hx_indicator="#spinner",
                hx_target="this",
                hx_swap="none",
                hx_boost="false",
                cls="text-green-300 hover:text-green-100 p-2 border-green-300 border-2 hover:border-green-100 w-full h-full",
            )
            if curr_keys
            else None,
            id="key-manage",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex flex-col md:flex-row justify-center gap-4 w-2/3",
        )

    ## layout
    def nav() -> fh.Nav:
        return fh.Nav(
            fh.A(
                f"{NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Svg(
                fh.NotStr(
                    """<style>
                        .spinner_zWVm { animation: spinner_5QiW 1.2s linear infinite, spinner_PnZo 1.2s linear infinite; }
                        .spinner_gfyD { animation: spinner_5QiW 1.2s linear infinite, spinner_4j7o 1.2s linear infinite; animation-delay: .1s; }
                        .spinner_T5JJ { animation: spinner_5QiW 1.2s linear infinite, spinner_fLK4 1.2s linear infinite; animation-delay: .1s; }
                        .spinner_E3Wz { animation: spinner_5QiW 1.2s linear infinite, spinner_tDji 1.2s linear infinite; animation-delay: .2s; }
                        .spinner_g2vs { animation: spinner_5QiW 1.2s linear infinite, spinner_CMiT 1.2s linear infinite; animation-delay: .2s; }
                        .spinner_ctYB { animation: spinner_5QiW 1.2s linear infinite, spinner_cHKR 1.2s linear infinite; animation-delay: .2s; }
                        .spinner_BDNj { animation: spinner_5QiW 1.2s linear infinite, spinner_Re6e 1.2s linear infinite; animation-delay: .3s; }
                        .spinner_rCw3 { animation: spinner_5QiW 1.2s linear infinite, spinner_EJmJ 1.2s linear infinite; animation-delay: .3s; }
                        .spinner_Rszm { animation: spinner_5QiW 1.2s linear infinite, spinner_YJOP 1.2s linear infinite; animation-delay: .4s; }
                        @keyframes spinner_5QiW { 0%, 50% { width: 7.33px; height: 7.33px; } 25% { width: 1.33px; height: 1.33px; } }
                        @keyframes spinner_PnZo { 0%, 50% { x: 1px; y: 1px; } 25% { x: 4px; y: 4px; } }
                        @keyframes spinner_4j7o { 0%, 50% { x: 8.33px; y: 1px; } 25% { x: 11.33px; y: 4px; } }
                        @keyframes spinner_fLK4 { 0%, 50% { x: 1px; y: 8.33px; } 25% { x: 4px; y: 11.33px; } }
                        @keyframes spinner_tDji { 0%, 50% { x: 15.66px; y: 1px; } 25% { x: 18.66px; y: 4px; } }
                        @keyframes spinner_CMiT { 0%, 50% { x: 8.33px; y: 8.33px; } 25% { x: 11.33px; y: 11.33px; } }
                        @keyframes spinner_cHKR { 0%, 50% { x: 1px; y: 15.66px; } 25% { x: 4px; y: 18.66px; } }
                        @keyframes spinner_Re6e { 0%, 50% { x: 15.66px; y: 8.33px; } 25% { x: 18.66px; y: 11.33px; } }
                        @keyframes spinner_EJmJ { 0%, 50% { x: 8.33px; y: 15.66px; } 25% { x: 11.33px; y: 18.66px; } }
                        @keyframes spinner_YJOP { 0%, 50% { x: 15.66px; y: 15.66px; } 25% { x: 18.66px; y: 18.66px; } }
                    </style>
                    <rect class="spinner_zWVm" x="1" y="1" width="7.33" height="7.33"/>
                    <rect class="spinner_gfyD" x="8.33" y="1" width="7.33" height="7.33"/>
                    <rect class="spinner_T5JJ" x="1" y="8.33" width="7.33" height="7.33"/>
                    <rect class="spinner_E3Wz" x="15.66" y="1" width="7.33" height="7.33"/>
                    <rect class="spinner_g2vs" x="8.33" y="8.33" width="7.33" height="7.33"/>
                    <rect class="spinner_ctYB" x="1" y="15.66" width="7.33" height="7.33"/>
                    <rect class="spinner_BDNj" x="15.66" y="8.33" width="7.33" height="7.33"/>
                    <rect class="spinner_rCw3" x="8.33" y="15.66" width="7.33" height="7.33"/>
                    <rect class="spinner_Rszm" x="15.66" y="15.66" width="7.33" height="7.33"/>
                    """
                ),
                width="24",
                height="24",
                viewBox="0 0 24 24",
                fill="none",
                id="spinner",
                cls="htmx-indicator w-8 h-8 absolute top-4 left-1/2 transform -translate-x-1/2 fill-blue-300",
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
            cls="flex justify-between p-4 relative",
            style="max-height: 10vh;",
        )

    def main_content(session: dict, db_session: DBSession) -> fh.Main:
        curr_gen_form = session["gen_form"]
        curr_gens = get_curr_gens(session, db_session)
        gen_containers = [gen_view(g, session, db_session) for g in curr_gens]
        return fh.Main(
            fh.Div(
                gen_form_toggle(curr_gen_form),
                fh.Div(
                    id="gen-form",
                    hx_get="/get-gen-form/" + curr_gen_form,
                    hx_indicator="#spinner",
                    hx_target="#gen-form",
                    hx_swap="outerHTML",
                    hx_trigger="load",
                ),
                cls="w-2/3 flex flex-col gap-4 justify-center items-center",
            ),
            gen_manage(curr_gens),
            fh.Div(
                *gen_containers[::-1],
                id="gen-list",
                cls="flex flex-col justify-center items-center gap-2 w-2/3",
                style="max-height: 40vh; overflow-y: auto;",
                hx_ext="sse",
                sse_connect="/stream-gens",
                hx_swap="afterbegin",
                sse_swap="message",
            ),
            cls="flex flex-col justify-center items-center gap-4 p-8",
            style="max-height: 80vh;",
        )

    def developer_page(session: dict, db_session: DBSession) -> fh.Main:
        curr_keys = get_curr_keys(session, db_session)
        read_keys = [ApiKeyRead.model_validate(k) for k in curr_keys]
        key_containers = [key_view(k, session, db_session) for k in read_keys]
        return fh.Main(
            fh.Button(
                "Request New Key",
                id="request-new-key",
                hx_post="/request-key",
                hx_indicator="#spinner",
                hx_target="#api-key-table",
                hx_swap="afterbegin",
                cls="text-blue-300 hover:text-blue-100 p-2 w-2/3 border-blue-300 border-2 hover:border-blue-100",
            ),
            key_manage(read_keys),
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

    def toast_container() -> fh.Div:
        return fh.Div(id="toast-container", cls="hidden")

    def footer(
        db_session: DBSession,
    ) -> fh.Footer:
        return fh.Footer(
            fh.Div(
                fh.Div(
                    balance_view(GlobalBalanceRead.model_validate(get_curr_balance(db_session))),
                    id="balance",
                    cls="flex items-start gap-0.5 md:gap-1",
                    hx_ext="sse",
                    sse_connect="/stream-balance",
                    hx_swap="innerHTML",
                    sse_swap="message",
                ),
                fh.P(
                    fh.A("Buy 50 more", href="/buy_global", cls="font-bold text-blue-300 hover:text-blue-100"),
                    " to share ($1)",
                ),
                cls="flex flex-col gap-0.5",
            ),
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

    # helper fns
    ## validation
    def validate_image_url(image_url: str) -> bool:
        return validators.url(image_url)

    def validate_image_file(image_file: fh.UploadFile = None, upload_path: Path = None) -> str | Path:
        if image_file is not None:
            # Ensure extension is valid image
            valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
            file_extension = Path(image_file.filename).suffix.lower()
            if file_extension not in valid_extensions:
                return "Invalid file type. Please upload an image."

            # Write file to disk
            filebuffer = image_file.read()
            upload_path = upload_dir / f"{uuid.uuid4()}{file_extension}"
            upload_path.write_bytes(filebuffer)

        # Verify upload path
        if not upload_path.exists():
            return "Error: File not found."

        # Verify MIME type and magic #
        img = Image.open(upload_path)
        try:
            img.verify()
        except Exception as e:
            os.remove(upload_path)
            return f"Error: {e}"

        # Limit img size
        MAX_FILE_SIZE_MB = 5
        MAX_DIMENSIONS = (4096, 4096)
        if os.path.getsize(upload_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
            os.remove(upload_path)
            return f"File size exceeds {MAX_FILE_SIZE_MB}MB limit."
        with Image.open(upload_path) as img:
            if img.size[0] > MAX_DIMENSIONS[0] or img.size[1] > MAX_DIMENSIONS[1]:
                os.remove(upload_path)
                return f"Image dimensions exceed {MAX_DIMENSIONS[0]}x{MAX_DIMENSIONS[1]} pixels limit."

        # Run antivirus
        try:
            result = subprocess.run(  # noqa: S603
                ["python", "main.py", str(upload_path)],  # noqa: S607
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/Python-Antivirus",
            )
            scan_result = result.stdout.strip().lower()
            if scan_result == "infected":
                os.remove(upload_path)
                return "Potential threat detected."
        except Exception as e:
            os.remove(upload_path)
            return f"Error during antivirus scan: {e}"

        return upload_path

    ## generation
    @fh.threaded
    def generate_and_save(g: GenCreate, session: dict, db_session: DBSession) -> None:
        k = ApiKeyCreate(session_id=session["session_id"])
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
        # db_session.add(g)
        # db_session.commit()
        # db_session.refresh(g)
        # return

        # TODO: uncomment for debugging
        # response = requests.Response()
        # response.status_code = 500

        if not response.ok:
            fh.add_toast(session, "Failed with status code: " + str(response.status_code), "error")
            g.failed = True
        else:
            g.response = response.json()
        db_session.add(g)
        db_session.commit()
        db_session.refresh(g)

    ## key generation
    def generate_key_and_save(k: ApiKeyCreate, db_session: DBSession) -> None:
        k.key = secrets.token_hex(32)
        db_session.add(k)
        db_session.commit()
        db_session.refresh(k)

    # SSE helpers
    async def stream_gen_updates(session: dict, db_session: DBSession):
        while not shutdown_event.is_set():
            curr_gens = get_curr_gens(session, db_session)
            for g in curr_gens:
                current_state = "response" if g.response else "failed" if g.failed else "loading"
                if g.id not in shown_generations or shown_generations[g.id] != current_state:
                    shown_generations[g.id] = current_state
                    yield fh.sse_message(fh.Script(f"document.getElementById('gen-{g.id}').remove();"))
                    yield fh.sse_message(gen_view(g, session))
            await sleep(1)

    async def stream_balance_updates(db_session: DBSession):
        while not shutdown_event.is_set():
            curr_balance = get_curr_balance(db_session).balance
            global shown_balance
            if shown_balance != curr_balance.balance:
                shown_balance = curr_balance.balance
                yield fh.sse_message(balance_view(GlobalBalanceRead.model_validate(curr_balance)))
            await sleep(1)

    # routes
    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## toasts without target
    @f_app.post("/toast")
    async def toast(session: dict, message: str, type: str) -> fh.Div:
        fh.add_toast(session, message, type)
        return fh.Div(id="toast-container", cls="hidden")

    ## pages
    @f_app.get("/")
    async def home(session: dict, db_session: DBSession = get_db_session()) -> tuple:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_hex(32)
        if "gen_form" not in session:
            session["gen_form"] = "image-url"
        return (
            fh.Title(NAME),
            fh.Div(
                nav(),
                main_content(session, db_session),
                toast_container(),
                footer(db_session),
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

    @f_app.get("/developer")
    def developer(session: dict, db_session: DBSession = get_db_session()) -> tuple:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_hex(32)
        return (
            fh.Title(NAME + " | " + "developer"),
            fh.Div(
                nav(),
                developer_page(session, db_session),
                toast_container(),
                footer(db_session),
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

    ## SSE streams
    @f_app.get("/stream-gens")
    async def stream_gens(session: dict, db_session: DBSession = get_db_session()):
        return fh.EventStream(stream_gen_updates(session, db_session))

    @f_app.get("/stream-balance")
    async def stream_balance(db_session: DBSession = get_db_session()):
        return fh.EventStream(stream_balance_updates(db_session))

    ## gen form view
    @f_app.get("/get-gen-form/{view}")
    def get_gen_form(view: str, session: dict) -> tuple:
        session["gen_form"] = view
        return (
            (
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
                    hx_indicator="#spinner",
                    hx_target="#gen-list",
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
                    hx_indicator="#spinner",
                    hx_target="#gen-list",
                    hx_swap="afterbegin",
                    id="gen-form",
                    cls="w-full h-full",
                ),
            ),
            gen_form_toggle(view, "outerHTML:#gen-form-toggle"),
        )

    ## generation routes
    @f_app.post("/url")
    def generate_from_url(
        session: dict, image_url: str, question: str, db_session: DBSession = get_db_session()
    ) -> tuple:
        # validation
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if not image_url:
            fh.add_toast(session, "No image URL provided", "error")
            return None
        if not validate_image_url(image_url):
            fh.add_toast(session, "Invalid image URL", "error")
            return None

        # Warn if we're out of balance
        curr_balance = get_curr_balance(db_session)
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.balance -= 1
        db_session.add(curr_balance)
        db_session.commit()
        db_session.refresh(curr_balance)

        # Clear input
        clear_img_input = fh.Input(
            id="new-image-url", name="image_url", placeholder="Enter an image url", hx_swap_oob="true"
        )
        clear_q_input = fh.Input(
            id="new-question", name="question", placeholder="Specify format or question", hx_swap_oob="true"
        )

        # Generate as before
        g = GenCreate(
            image_url=image_url,
            question=question,
            session_id=session["session_id"],
        )
        generate_and_save(g, session, db_session)
        g_read = GenRead.model_validate(g)
        return (
            gen_view(g_read, session, db_session),
            clear_img_input,
            clear_q_input,
            gen_manage(get_curr_gens(session, db_session), "outerHTML:#gen-manage"),
        )

    @f_app.post("/upload")
    async def generate_from_upload(
        session: dict, image_file: fh.UploadFile, question: str, db_session: DBSession
    ) -> tuple:
        # Check for session ID
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        # Validate CSRF token
        if "csrf_token" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        # Check for empty file
        if not image_file:
            fh.add_toast(session, "No image uploaded", "error")
            return None

        # Validate image file
        res = validate_image_file(image_file)
        if isinstance(res, str):
            fh.add_toast(session, res, "error")
            return None
        else:
            upload_path = res

        # Warn if we're out of balance
        curr_balance = get_curr_balance(db_session)
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.balance -= 1
        db_session.add(curr_balance)
        db_session.commit()
        db_session.refresh(curr_balance)

        # Clear input
        clear_img_input = fh.Input(
            id="new-image-upload", name="image_file", type="file", accept="image/*", hx_swap_oob="true"
        )
        clear_q_input = fh.Input(
            id="new-question", name="question", placeholder="Specify format or question", hx_swap_oob="true"
        )

        # Generate as before
        g = GenCreate(
            image_file=upload_path,
            question=question,
            session_id=session["session_id"],
        )
        generate_and_save(g, session, db_session)
        g_read = GenRead.model_validate(g)
        return (
            gen_view(g_read, session, db_session),
            clear_img_input,
            clear_q_input,
            gen_manage(get_curr_gens(session, db_session), "outerHTML:#gen-manage"),
        )

    ## api key request
    @f_app.post("/request-key")
    def generate_key(session: dict, db_session: DBSession = get_db_session()) -> tuple:
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        k = ApiKeyCreate(session_id=session["session_id"])
        generate_key_and_save(k)
        k_read = ApiKeyRead.model_validate(k)
        read_keys = [ApiKeyRead.model_validate(k) for k in get_curr_keys(session, db_session)]
        return key_view(k_read, session, db_session), key_manage(read_keys, "outerHTML:#key-manage")

    ## clear
    @f_app.delete("/gens")
    def clear_all(session: dict, db_session: DBSession = get_db_session()) -> fh.RedirectResponse:
        ids = [g.id for g in get_curr_gens(session, db_session)]
        for id in ids:
            g = db_session.exec(select(Gen).where(Gen.id == id)).first()
            if g and g.image_file and os.path.exists(g.image_file):
                os.remove(g.image_file)
            db_session.delete(g)
            db_session.commit()
        fh.add_toast(session, "Deleted generations and image files.", "success")
        return fh.RedirectResponse("/", status_code=303)

    @f_app.delete("/keys")
    def clear_keys(session: dict, db_session: DBSession = get_db_session()) -> fh.RedirectResponse:
        ids = [k.id for k in get_curr_keys(session, db_session)]
        for id in ids:
            k = db_session.exec(select(ApiKey).where(ApiKey.id == id)).first()
            db_session.delete(k)
            db_session.commit()
        fh.add_toast(session, "Deleted keys.", "success")
        return fh.RedirectResponse("/developer", status_code=303)

    ## export to CSV
    @f_app.get("/export-gens")
    async def export_gens(req, db_session: DBSession = get_db_session()) -> fh.Response:
        session = req.session
        curr_gens = get_curr_gens(session, db_session)
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

    @f_app.get("/export-keys")
    async def export_keys(req, db_session: DBSession = get_db_session()) -> fh.Response:
        session = req.session
        curr_keys = get_curr_keys(session, db_session)
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
    def buy_credits(session: dict) -> fh.RedirectResponse:
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
    def cancel() -> fh.RedirectResponse:
        return fh.RedirectResponse("/")

    ### STRIPE sends the USER here after a payment was 'successful'.
    @f_app.get("/success")
    def success() -> fh.RedirectResponse:
        return fh.RedirectResponse("/")

    ### STRIPE calls this to tell APP when a payment was completed.
    @f_app.post("/webhook")
    async def stripe_webhook(request, db_session: DBSession = get_db_session()) -> dict:
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
            curr_balance = get_curr_balance(db_session)
            curr_balance.balance += 50
            db_session.add(curr_balance)
            db_session.commit()
            db_session.refresh(curr_balance)
            return {"status": "success"}, 200

    return f_app


# TODO:
# - add gens/keys counts: https://hypermedia.systems/more-htmx-patterns/#_lazy_loading
# - add granular delete: https://hypermedia.systems/more-htmx-patterns/#_inline_delete
# - add bulk delete: https://hypermedia.systems/more-htmx-patterns/#_bulk_delete
# - add better infinite scroll: https://hypermedia.systems/htmx-patterns/#_another_application_improvement_paging
# - add multiple file urls/uploads: https://docs.fastht.ml/tutorials/quickstart_for_web_devs.html#multiple-file-uploads

# - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#   - Only allow authorized users to upload files:
#       - add user authentication: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
#       - add form validation: https://hypermedia.systems/htmx-patterns/#_next_steps_validating_contact_emails
# - better url/file validation: https://hypermedia.systems/htmx-patterns/#_next_steps_validating_contact_emails
# - add animations: https://hypermedia.systems/a-dynamic-archive-ui/#_smoothing_things_out_animations_in_htmx
