import os
import secrets
from pathlib import Path

import modal

from utils import (
    DATA_VOLUME,
    IN_PROD,
    MINUTES,
    NAME,
    PYTHON_VERSION,
    REMOTE_DB_URI,
    VOLUME_CONFIG,
)

parent_path: Path = Path(__file__).parent


# Modal
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
        "sqlmodel==0.0.22",
    )
    .copy_local_file(parent_path / "favicon.ico", "/root/favicon.ico")
    .copy_local_dir(parent_path.parent / "db", "/root/db")
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
    from contextlib import contextmanager

    import requests
    import stripe
    import validators
    from fasthtml import common as fh
    from PIL import Image
    from simpleicons.icons import si_github, si_pypi
    from sqlmodel import Session as DBSession
    from sqlmodel import create_engine, select
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
        init_balance,
    )

    # setup
    def before(req, sess):
        req.scope["session_id"] = sess.setdefault("session_id", str(uuid.uuid4()))
        req.scope["csrf_token"] = sess.setdefault("csrf_token", secrets.token_hex(32))
        req.scope["gen_form"] = sess.setdefault("gen_form", "image-url")

    def _not_found(req, exc):
        message = "Page not found!"
        typing_steps = len(message)
        return (
            fh.Title(NAME + " | 404"),
            fh.Div(
                nav(),
                fh.Titled(
                    fh.P(
                        message,
                        hx_indicator="#spinner",
                        cls="text-2xl text-red-300 animate-typing overflow-hidden whitespace-nowrap border-r-4 border-red-300",
                        style=f"animation: typing 2s steps({typing_steps}, end), blink-caret .75s step-end infinite",
                    ),
                    cls="flex flex-col justify-center items-center gap-4 p-8",
                ),
                toast_container(),
                footer(),
                cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
            ),
            fh.Style(
                """
            @keyframes typing {
                from { width: 0; }
                to { width: 100%; }
            }
            @keyframes blink-caret {
                from, to { border-color: transparent; }
                50% { border-color: red; }
            }
            """
            ),
        )

    f_app, _ = fh.fast_app(
        ws_hdr=True,
        before=fh.Beforeware(before, skip=[r"/favicon\.ico", r"/static/.*", r".*\.css"]),
        exception_handlers={404: _not_found},
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

    engine = create_engine(
        url=REMOTE_DB_URI,
        # echo=not IN_PROD,
    )

    @contextmanager
    def get_db_session():
        with DBSession(engine) as session:
            yield session

    def get_curr_gens(
        session_id,
        number: int = None,
        offset: int = 0,
    ) -> list[Gen]:
        with get_db_session() as db_session:
            query = select(Gen).where(Gen.session_id == session_id).order_by(Gen.request_at.desc()).offset(offset)
            if number:
                query = query.limit(number)
            return db_session.exec(query).all()

    def get_curr_keys(
        session_id,
        number: int = None,
        offset: int = 0,
    ) -> list[ApiKey]:
        with get_db_session() as db_session:
            query = (
                select(ApiKey).where(ApiKey.session_id == session_id).order_by(ApiKey.granted_at.desc()).offset(offset)
            )
            if number:
                query = query.limit(number)
            return db_session.exec(query).all()

    def get_curr_balance() -> GlobalBalance:
        with get_db_session() as db_session:
            curr_balance = db_session.get(GlobalBalance, 1)
            if not curr_balance:
                new_balance = GlobalBalanceCreate(balance=init_balance)
                curr_balance = GlobalBalance.model_validate(new_balance)
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return curr_balance

    ## stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    DOMAIN: str = os.environ["DOMAIN"]

    ## SSE state
    shutdown_event = fh.signal_shutdown()
    global shown_generations
    shown_generations = {}
    global shown_balance
    shown_balance = 0

    ## pagination
    max_gens = 10
    max_keys = 20

    # ui
    ## components
    def gen_view(
        g: GenRead,
        session,
    ):
        ### check if g is valid
        with get_db_session() as db_session:
            if db_session.get(Gen, g.id) is None:
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

        limit_chars = 100
        if g.failed:
            return fh.Card(
                fh.Div(
                    fh.Img(
                        src=image_src,
                        alt="Card image",
                        cls="max-h-48 w-full object-contain",
                    ),
                    cls="w-1/2",
                ),
                fh.Div(
                    fh.P(
                        g.question[:limit_chars] + ("..." if len(g.question) > limit_chars else ""),
                        onclick=f"navigator.clipboard.writeText('{g.question}');",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_indicator="#spinner",
                        hx_target="#toast-container",
                        hx_swap="outerHTML",
                        cls="text-blue-300 hover:text-blue-100 cursor-pointer max-w-full",
                        title="Click to copy",
                    ),
                    fh.P(
                        "Generation failed",
                        cls="text-red-300",
                    ),
                    cls="max-h-48 w-1/2 flex flex-col gap-2",
                ),
                cls="max-h-60 w-full flex justify-between gap-4",
                id=f"gen-{g.id}",
            )
        elif g.response:
            return fh.Card(
                fh.Div(
                    fh.Img(
                        src=image_src,
                        alt="Card image",
                        cls="max-h-48 w-full object-contain",
                    ),
                    cls="w-1/2",
                ),
                fh.Div(
                    fh.P(
                        g.question[:limit_chars] + ("..." if len(g.question) > limit_chars else ""),
                        onclick=f"navigator.clipboard.writeText('{g.question}');",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_indicator="#spinner",
                        hx_target="#toast-container",
                        hx_swap="outerHTML",
                        cls="text-blue-300 hover:text-blue-100 cursor-pointer max-w-full",
                        title="Click to copy",
                    ),
                    fh.P(
                        g.response[:limit_chars] + ("..." if len(g.response) > limit_chars else ""),
                        onclick=f"navigator.clipboard.writeText('{g.response}');",
                        hx_post="/toast?message=Copied to clipboard!&type=success",
                        hx_indicator="#spinner",
                        hx_target="#toast-container",
                        hx_swap="outerHTML",
                        cls="text-green-300 hover:text-green-100 cursor-pointer max-w-full",
                        title="Click to copy",
                    ),
                    cls="max-h-48 w-1/2 flex flex-col gap-2",
                ),
                cls="max-h-60 w-full flex justify-between gap-4",
                id=f"gen-{g.id}",
            )
        return fh.Card(
            fh.Div(
                fh.Img(
                    src=image_src,
                    alt="Card image",
                    cls="max-h-48 w-full object-contain",
                ),
                cls="w-1/2",
            ),
            fh.Div(
                fh.P(
                    g.question[:limit_chars] + ("..." if len(g.question) > limit_chars else ""),
                    onclick=f"navigator.clipboard.writeText('{g.question}');",
                    hx_post="/toast?message=Copied to clipboard!&type=success",
                    hx_indicator="#spinner",
                    hx_target="#toast-container",
                    hx_swap="outerHTML",
                    cls="text-blue-300 hover:text-blue-100 cursor-pointer max-w-full",
                    title="Click to copy",
                ),
                fh.P("Scanning image ..."),
                cls="max-h-48 w-1/2 flex flex-col gap-2",
            ),
            cls="max-h-60 w-full flex justify-between gap-4",
            id=f"gen-{g.id}",
        )

    def key_view(
        k: ApiKeyRead,
        session,
    ):
        with get_db_session() as db_session:
            if db_session.get(ApiKey, k.id) is None:
                return None
        if k.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        if k.key and k.granted_at:
            obscured_key = k.key[:4] + "*" * (len(k.key) - 4)
            short_key = obscured_key[:8] + "..."

            return (
                fh.Div(
                    fh.Button(
                        fh.Svg(
                            """<path d="M10 12L14 16M14 12L10 16M18 6L17.1991 18.0129C17.129 19.065 17.0939 19.5911 16.8667 19.99C16.6666 20.3412 16.3648 20.6235 16.0011 20.7998C15.588 21 15.0607 21 14.0062 21H9.99377C8.93927 21 8.41202 21 7.99889 20.7998C7.63517 20.6235 7.33339 20.3412 7.13332 19.99C6.90607 19.5911 6.871 19.065 6.80086 18.0129L6 6M4 6H20M16 6L15.7294 5.18807C15.4671 4.40125 15.3359 4.00784 15.0927 3.71698C14.8779 3.46013 14.6021 3.26132 14.2905 3.13878C13.9376 3 13.523 3 12.6936 3H11.3064C10.477 3 10.0624 3 9.70951 3.13878C9.39792 3.26132 9.12208 3.46013 8.90729 3.71698C8.66405 4.00784 8.53292 4.40125 8.27064 5.18807L8 6" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>""",
                            width="24",
                            height="24",
                            viewBox="0 0 24 24",
                            fill="none",
                            cls="htmx-indicator w-8 h-8 fill-red-300 hover:fill-red-100",
                        ),
                        hx_delete=f"/key/{k.id}",
                        hx_indicator="#spinner",
                        hx_target="closest div",
                        hx_swap="outerHTML swap:1s",
                        hx_confirm="Are you sure?",
                        cls="w-1/8",
                    ),
                    fh.Style(
                        """
                        div.htmx-swapping {
                            opacity: 0;
                            transition: opacity 1s ease-out;
                        }
                        """
                    ),
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
                        cls="text-blue-300 hover:text-blue-100 cursor-pointer w-5/8",
                        title="Click to copy",
                        id=f"key-element-{k.id}",
                    ),
                    fh.Div(
                        k.granted_at.strftime("%Y-%m-%d %H:%M:%S"),
                        cls="w-3/8",
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

    def balance_view(
        gb: GlobalBalanceRead,
    ):
        with get_db_session() as db_session:
            if db_session.get(GlobalBalance, gb.id) is None:
                return None

        return (
            fh.P("Global balance:"),
            fh.P(f"{gb.balance} credits", cls="font-bold"),
        )

    def gen_form_toggle(gen_form: str, hx_swap_oob: bool = "false"):
        return fh.Div(
            fh.Button(
                "Image URL",
                id="gen-form-toggle-url",
                cls="w-full h-full text-blue-100 bg-blue-500 p-2 border-blue-500 border-2"
                if gen_form == "image-url"
                else "w-full h-full text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                hx_get="/get-gen-form?view=image-url",
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
                hx_get="/get-gen-form?view=image-upload",
                hx_indicator="#spinner",
                hx_target="#gen-form",
                hx_swap="innerHTML",
            ),
            id="gen-form-toggle",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-full flex flex-col md:flex-row gap-2 md:gap-4",
        )

    def num_gens(gen_count, hx_swap_oob: bool = "false"):
        return fh.Div(
            fh.P(
                f"({gen_count} total generations)",
                hx_ext="sse",
                sse_connect="/stream-gen-count",
                sse_swap="UpdateGensCount",
                cls="text-blue-300 text-md whitespace-nowrap",
            ),
            id="gen-count",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-auto h-full flex justify-center",
        )

    def num_keys(key_count, hx_swap_oob: bool = "false"):
        return fh.Div(
            fh.P(
                f"({key_count} total keys)",
                cls="text-blue-300 text-md whitespace-nowrap",
            ),
            id="key-count",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-auto h-full",
        )

    def gen_manage(gens_present: bool, hx_swap_oob: bool = "false"):
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
            if gens_present
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
            if gens_present
            else None,
            id="gen-manage",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex flex-col md:flex-row justify-center gap-2 md:gap-4 w-full md:w-2/3",
        )

    def key_manage(keys_present: bool, hx_swap_oob: bool = "false"):
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
            if keys_present
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
            if keys_present
            else None,
            id="key-manage",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="flex flex-col md:flex-row justify-center gap-4 w-full md:w-2/3",
        )

    def gen_load_more(gens_present: bool, still_more: bool, idx: int = 2, hx_swap_oob: bool = "false"):
        return fh.Div(
            fh.Button(
                "Load More",
                hx_get=f"/page-gens?idx={idx}",
                hx_indicator="#spinner",
                hx_target="#gen-list",
                hx_swap="beforeend",
                cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100 w-full h-full",
            )
            if gens_present and still_more
            else None,
            id="load-more-gens",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-full md:w-2/3",
        )

    def key_load_more(keys_present: bool, still_more: bool, idx: int = 2, hx_swap_oob: bool = "false"):
        return fh.Div(
            fh.Button(
                "Load More",
                hx_get=f"/page-keys?idx={idx}",
                hx_indicator="#spinner",
                hx_target="#api-key-table",
                hx_swap="beforeend",
                cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100 w-full h-full",
            )
            if keys_present and still_more
            else None,
            id="load-more-keys",
            hx_swap_oob=hx_swap_oob if hx_swap_oob != "false" else None,
            cls="w-full md:w-2/3",
        )

    ## layout
    def nav():
        return fh.Nav(
            fh.A(
                f"{NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Svg(
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
                """,
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
                        fh.Svg(
                            si_github.svg,
                            width="35",
                            height="35",
                            viewBox="0 0 15 15",
                            fill="none",
                            cls="rounded p-0.5 hover:bg-zinc-700 cursor-pointer",
                        ),
                        href="https://github.com/andrewhinh/formless",
                        target="_blank",
                    ),
                    fh.A(
                        fh.Svg(
                            si_pypi.svg,
                            width="35",
                            height="35",
                            viewBox="0 0 15 15",
                            fill="none",
                            cls="rounded p-0.5 hover:bg-zinc-700 cursor-pointer",
                        ),
                        href="https://pypi.org/project/formless/",
                        target="_blank",
                    ),
                    cls="flex flex-row gap-4",
                ),
                cls="flex flex-col items-end md:flex-row md:items-center gap-2 md:gap-8",
            ),
            cls="flex justify-between p-4 relative",
        )

    def main_content(
        session,
    ):
        curr_gen_form = session["gen_form"]
        gens_present = bool(get_curr_gens(session["session_id"], number=1))
        return fh.Main(
            fh.Div(
                gen_form_toggle(curr_gen_form),
                fh.Div(
                    id="gen-form",
                    hx_get=f"/get-gen-form?view={curr_gen_form}",
                    hx_indicator="#spinner",
                    hx_target="#gen-form",
                    hx_swap="outerHTML",
                    hx_trigger="load",
                ),
                cls="w-full md:w-2/3 flex flex-col gap-4 justify-center items-center",
            ),
            num_gens(len(get_curr_gens(session["session_id"]))),
            gen_manage(gens_present),
            fh.Div(
                get_gen_table_part(session),
                id="gen-list",
                cls="flex flex-col justify-center items-center gap-2 w-full md:w-2/3",
                hx_ext="sse",
                sse_connect="/stream-gens",
                sse_swap="UpdateGens",
            ),
            gen_load_more(
                gens_present, len(get_curr_gens(session["session_id"], number=max_gens, offset=max_gens)) > 0
            ),
            cls="flex flex-col justify-start items-center grow gap-4 p-8",
        )

    def developer_page(
        session,
    ):
        keys_present = bool(get_curr_keys(session["session_id"], number=1))
        return fh.Main(
            fh.Div(
                fh.Button(
                    "Request New Key",
                    id="request-new-key",
                    hx_post="/request-key",
                    hx_indicator="#spinner",
                    hx_target="#api-key-table",
                    hx_swap="afterbegin",
                    cls="text-blue-300 hover:text-blue-100 p-2 w-full border-blue-300 border-2 hover:border-blue-100",
                ),
                num_keys(len(get_curr_keys(session["session_id"]))),
                cls="w-full md:w-2/3 flex gap-4 justify-center items-center",
            ),
            key_manage(keys_present),
            fh.Div(
                fh.Div(
                    fh.Div("Key", cls="font-bold w-2/3"),
                    fh.Div("Granted At", cls="font-bold w-1/3"),
                    cls="flex p-2",
                ),
                fh.Div(
                    get_key_table_part(session),
                    id="api-key-table",
                ),
                cls="w-full md:w-2/3 flex flex-col gap-2 text-sm md:text-lg border-slate-500 border-2",
            ),
            key_load_more(
                keys_present, len(get_curr_keys(session["session_id"], number=max_keys, offset=max_keys)) > 0
            ),
            cls="flex flex-col justify-start items-center grow gap-4 p-8",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            fh.Div(
                fh.Div(
                    balance_view(GlobalBalanceRead.model_validate(get_curr_balance())),
                    id="balance",
                    cls="flex items-start gap-0.5 md:gap-1",
                    hx_ext="sse",
                    sse_connect="/stream-balance",
                    sse_swap="UpdateBalance",
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
            filebuffer = image_file.file.read()
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
    def generate_and_save(
        g: Gen,
        session,
    ):
        k = ApiKeyCreate(session_id=session["session_id"])
        k = generate_key_and_save(k)

        # TODO: uncomment for debugging
        # g.sqlmodel_update({"response": "temp"})  # have to use sqlmodel_update since object is already committed
        # with get_db_session() as db_session:
        #     db_session.add(g)
        #     db_session.commit()
        #     db_session.refresh(g)
        #     return

        # TODO: uncomment for debugging
        # g.sqlmodel_update({"failed": True})
        # with get_db_session() as db_session:
        #     db_session.add(g)
        #     db_session.commit()
        #     db_session.refresh(g)
        #     return

        if g.image_url:
            response = requests.post(
                os.getenv("API_URL"),
                json={"image_url": g.image_url, "question": g.question},
                headers={"X-API-Key": k.key},
            )
        elif g.image_file:
            response = requests.post(
                f"{os.getenv('API_URL')}/upload",
                files={"image": open(g.image_file, "rb")},
                data={"question": g.question},
                headers={
                    "X-API-Key": k.key,
                },
            )

        if not response.ok:
            fh.add_toast(session, "Failed with status code: " + str(response.status_code), "error")
            g.sqlmodel_update({"failed": True})
        else:
            g.sqlmodel_update({"response": response.json()})
        with get_db_session() as db_session:
            db_session.add(g)
            db_session.commit()
            db_session.refresh(g)

    def generate_key_and_save(
        k: ApiKeyCreate,
    ) -> ApiKey:
        k.key = secrets.token_hex(16)
        k = ApiKey.model_validate(k)
        with get_db_session() as db_session:
            db_session.add(k)
            db_session.commit()
            db_session.refresh(k)
        return k

    ## SSE helpers
    async def stream_gen_updates(
        session,
    ):
        while not shutdown_event.is_set():
            global shown_generations
            shown_ids = list(shown_generations.keys())
            if not shown_ids:
                yield fh.sse_message(get_gen_table_part(session))

            curr_gens = get_curr_gens(session["session_id"], number=len(shown_ids), offset=min(shown_ids, default=0))
            read_gens = [GenRead.model_validate(g) for g in curr_gens]
            inner_content = ""
            updated = False
            for g in read_gens:
                current_state = "response" if g.response else "failed" if g.failed else "loading"
                if shown_generations.get(g.id) != current_state:
                    shown_generations[g.id] = current_state
                    updated = True
                inner_content += str(gen_view(g, session))
            if updated:
                yield fh.sse_message(fh.NotStr(inner_content[::-1]))
            await sleep(1)

    async def stream_gen_count_updates(
        session,
    ):
        while not shutdown_event.is_set():
            curr_gens = get_curr_gens(session["session_id"])
            if len(curr_gens) != len(shown_generations):
                yield fh.sse_message(num_gens(len(curr_gens)))
            await sleep(1)

    async def stream_balance_updates():
        while not shutdown_event.is_set():
            curr_balance = get_curr_balance()
            global shown_balance
            if shown_balance != curr_balance.balance:
                shown_balance = curr_balance.balance
                yield fh.sse_message(balance_view(GlobalBalanceRead.model_validate(curr_balance)))
            await sleep(1)

    ## pagination
    def get_gen_table_part(session, part_num: int = 1, size: int = max_gens):
        curr_gens = get_curr_gens(session["session_id"], number=size, offset=(part_num - 1) * size)
        read_gens = [GenRead.model_validate(g) for g in curr_gens]
        paginated = [gen_view(g, session) for g in read_gens]
        return tuple(paginated)

    def get_key_table_part(session, part_num: int = 1, size: int = max_keys):
        curr_keys = get_curr_keys(session["session_id"], number=size, offset=(part_num - 1) * size)
        read_keys = [ApiKeyRead.model_validate(k) for k in curr_keys]
        paginated = [key_view(k, session) for k in read_keys]
        return tuple(paginated)

    # routes
    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## toasts without target
    @f_app.post("/toast")
    def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return fh.Div(id="toast-container", cls="hidden")

    ## pages
    @f_app.get("/")
    def home(
        session,
    ):
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

    @f_app.get("/developer")
    def developer(
        session,
    ):
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

    ## SSE streams
    @f_app.get("/stream-gens")
    async def stream_gens(
        session,
    ):
        return fh.EventStream(stream_gen_updates(session))

    @f_app.get("/stream-gen-count")
    async def stream_gen_count(
        session,
    ):
        return fh.EventStream(stream_gen_count_updates(session))

    @f_app.get("/stream-balance")
    async def stream_balance():
        return fh.EventStream(stream_balance_updates())

    ## gen form view
    @f_app.get("/get-gen-form")
    def get_gen_form(session, view: str):
        session["gen_form"] = view
        return (
            (
                fh.Form(
                    fh.Div(
                        fh.Input(
                            id="new-image-url",
                            name="image_url",  # passed to fn call for python syntax
                            placeholder="Enter an image url",
                            hx_target="this",
                            hx_swap="outerHTML",
                            hx_trigger="change, keyup delay:200ms changed",
                            hx_post="/check-url",
                            hx_indicator="#spinner",
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
                            hx_target="this",
                            hx_swap="none",
                            hx_trigger="change delay:200ms changed",
                            hx_post="/check-upload",
                            hx_indicator="#spinner",
                            hx_encoding="multipart/form-data",  # correct file encoding for check-upload since not in form
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
            gen_form_toggle(view, "true"),
        )

    ## input validation
    @f_app.post("/check-url")
    def check_url(session, image_url: str):
        if not validate_image_url(image_url):
            fh.add_toast(session, "Invalid image URL", "error")
        return (
            fh.Input(
                value=image_url,
                id="new-image-url",
                name="image_url",
                placeholder="Enter an image url",
                hx_target="this",
                hx_swap="outerHTML",
                hx_trigger="change, keyup delay:200ms changed",
                hx_post="/check-url",
            ),
        )

    @f_app.post("/check-upload")
    def check_upload(
        session,
        image_file: fh.UploadFile,
    ):
        res = validate_image_file(image_file)
        if isinstance(res, str):
            fh.add_toast(session, res, "error")
        return fh.Div(cls="hidden")

    ## pagination
    @f_app.get("/page-gens")
    def page_gens(session, idx: int):
        return get_gen_table_part(session, idx), gen_load_more(
            bool(get_curr_gens(session["session_id"], number=1)),
            len(get_curr_gens(session["session_id"], number=max_gens, offset=max_gens * (idx + 1))) > 0,
            idx + 1,
            "true",
        )

    @f_app.get("/page-keys")
    def page_keys(session, idx: int):
        return get_key_table_part(session, idx), key_load_more(
            bool(get_curr_keys(session["session_id"], number=1)),
            len(get_curr_keys(session["session_id"], number=max_keys, offset=max_keys * (idx + 1))) > 0,
            idx + 1,
            "true",
        )

    ## generation routes
    @f_app.post("/url")
    def generate_from_url(
        session,
        image_url: str,
        question: str,
    ):
        # validation
        if not image_url:
            fh.add_toast(session, "No image URL provided", "error")
            return None
        if not validate_image_url(image_url):
            fh.add_toast(session, "Invalid image URL", "error")
            return None

        # Warn if we're out of balance
        curr_balance = get_curr_balance()
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.sqlmodel_update({"balance": curr_balance.balance - 1})
        with get_db_session() as db_session:
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
        ## need to put in db since generate_and_save is threaded
        g = Gen.model_validate(g)
        with get_db_session() as db_session:
            db_session.add(g)
            db_session.commit()
            db_session.refresh(g)
        generate_and_save(g, session)
        g_read = GenRead.model_validate(g)
        gens_present = bool(get_curr_gens(session["session_id"], number=1))
        return (
            gen_view(g_read, session),
            clear_img_input,
            clear_q_input,
            num_gens(len(get_curr_gens(session["session_id"])), "true"),
            gen_manage(gens_present, "true"),
            gen_load_more(
                gens_present,
                False,  # TODO: fix pagination when new results are added
                hx_swap_oob="true",
            ),
        )

    @f_app.post("/upload")
    def generate_from_upload(
        session,
        image_file: fh.UploadFile,
        question: str,
    ):
        if not image_file:
            fh.add_toast(session, "No image uploaded", "error")
            return None
        res = validate_image_file(image_file)
        if isinstance(res, str):
            fh.add_toast(session, res, "error")
            return None
        else:
            upload_path = res
        curr_balance = get_curr_balance()
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        curr_balance.sqlmodel_update({"balance": curr_balance.balance - 1})
        with get_db_session() as db_session:
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
            image_file=str(upload_path),
            question=question,
            session_id=session["session_id"],
        )
        ## need to put in db since generate_and_save is threaded
        g = Gen.model_validate(g)
        with get_db_session() as db_session:
            db_session.add(g)
            db_session.commit()
            db_session.refresh(g)
        generate_and_save(g, session)
        g_read = GenRead.model_validate(g)
        gens_present = bool(get_curr_gens(session["session_id"], number=1))
        return (
            gen_view(g_read, session),
            clear_img_input,
            clear_q_input,
            num_gens(len(get_curr_gens(session["session_id"])), "true"),
            gen_manage(gens_present, "true"),
            gen_load_more(
                gens_present,
                False,
                hx_swap_oob="true",
            ),
        )

    ## api key request
    @f_app.post("/request-key")
    def generate_key(
        session,
    ):
        k = ApiKeyCreate(session_id=session["session_id"])
        k = generate_key_and_save(k)
        k_read = ApiKeyRead.model_validate(k)
        keys_present = bool(get_curr_keys(session["session_id"], number=1))
        return (
            key_view(k_read, session),
            num_keys(len(get_curr_keys(session["session_id"])), "true"),
            key_manage(
                keys_present,
                "true",
            ),
            key_load_more(
                keys_present,
                False,
                hx_swap_oob="true",
            ),
        )

    ## clear
    @f_app.delete("/gens")
    def clear_all(
        session,
    ):
        gens = get_curr_gens(session["session_id"])
        for g in gens:
            if g and g.image_file and os.path.exists(g.image_file):
                os.remove(g.image_file)
            with get_db_session() as db_session:
                db_session.delete(g)
                db_session.commit()
        fh.add_toast(session, "Deleted generations and image files.", "success")
        return fh.RedirectResponse("/", status_code=303)

    @f_app.delete("/keys")
    def clear_keys(
        session,
    ):
        keys = get_curr_keys(session["session_id"])
        for k in keys:
            with get_db_session() as db_session:
                db_session.delete(k)
                db_session.commit()
        fh.add_toast(session, "Deleted keys.", "success")
        return fh.RedirectResponse("/developer", status_code=303)

    @f_app.delete("/key/{key_id}")
    def delete_key(
        session,
        key_id: int,
    ):
        with get_db_session() as db_session:
            key = db_session.get(ApiKey, key_id)
            db_session.delete(key)
            db_session.commit()
        fh.add_toast(session, "Deleted key.", "success")
        keys_present = bool(get_curr_keys(session["session_id"], number=1))
        return (
            "",
            num_keys(len(get_curr_keys(session["session_id"])), "true"),
            key_manage(
                keys_present,
                "true",
            ),
            key_load_more(
                keys_present,
                False,
                hx_swap_oob="true",
            ),
        )

    ## export to CSV
    @f_app.get("/export-gens")
    def export_gens(
        req,
    ):
        session = req.session
        curr_gens = get_curr_gens(session["session_id"])
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
    def export_keys(
        req,
    ):
        session = req.session
        curr_keys = get_curr_keys(session["session_id"])
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
    def buy_credits():
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
    async def stripe_webhook(
        request,
    ):
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
            curr_balance.sqlmodel_update({"balance": curr_balance.balance + 50})
            with get_db_session() as db_session:
                db_session.add(curr_balance)
                db_session.commit()
                db_session.refresh(curr_balance)
            return {"status": "success"}, 200

    return f_app


# TODO:
# - add granular delete: https://hypermedia.systems/more-htmx-patterns/#_inline_delete
# - add bulk delete: https://hypermedia.systems/more-htmx-patterns/#_bulk_delete
# - add multiple file urls/uploads: https://docs.fastht.ml/tutorials/quickstart_for_web_devs.html#multiple-file-uploads

# - complete file upload security: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
#   - Only allow authorized users to upload files:
#       - add user authentication: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
