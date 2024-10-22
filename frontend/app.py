import os
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
        "python-fasthtml==0.6.10", "simpleicons==7.21.0", "requests==2.32.3", "sqlite-utils==3.18", "stripe==11.1.0"
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
    import time
    import uuid

    import requests
    import stripe
    from fasthtml import common as fh
    from simpleicons.icons import si_github, si_pypi

    # Setup
    ## FastHTML
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

    ## database
    db_path = f"/{DATA_VOLUME}/main.db"
    # TODO: uncomment for debugging
    # os.remove(db_path)
    tables = fh.database(db_path).t
    gens = tables.gens
    if gens not in tables:
        gens.create(image_url=str, response=str, session_id=str, id=int, pk="id")
    Generation = gens.dataclass()

    ## global balance
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

    ## preview while waiting for response
    def generation_preview(g, session):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None
        if g.session_id != session["session_id"]:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        if g.response:
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

    ## api key request preview
    def key_request_preview(key, session):
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        if key["key"] and key["granted_at"]:
            return fh.Tr(
                fh.Td(key["key"]),
                fh.Td(key["granted_at"]),
                id=f"key-{key['key']}",
            )
        return fh.Tr(
            fh.Td("Requesting new key ..."),
            fh.Td(""),
            id=f"key-{key['key']}",
            hx_get=f"/keys/{key['key']}",
            hx_trigger="every 2s",
            hx_swap="outerHTML",
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
                    "Developer",
                    href="/developer",
                    cls="text-lg text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
                ),
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
                cls="flex flex-col items-end md:flex-row md:items-center gap-4 md:gap-8",
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
                    fh.Button(
                        "Scan",
                        cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                    ),
                ),
                hx_post="/",
                target_id="gen-list",
                hx_swap="afterbegin",
                cls="w-2/3",
            ),
            fh.Button(
                "Clear all",
                id="clear-gens",
                hx_post="/clear-gens",
                target_id="gen-list",
                hx_swap="innerHTML",
                cls=f"text-red-300 hover:text-red-100 p-2 w-1/3 border-red-300 border-2 hover:border-red-100 {'hidden' if not gen_containers else ''}",
            ),
            # fh.Div(
            #     fh.Button(
            #         "Clear all",
            #         id="clear-all",
            #         hx_post="/clear",
            #         target_id="gen-list",
            #         hx_swap="innerHTML",
            #         cls=f"text-red-300 hover:text-red-100 p-2 w-full border-red-300 border-2 hover:border-red-100 {'hidden' if not gen_containers else ''}",
            #     ),
            #     fh.Button(
            #         "Export to CSV",
            #         id="export-csv",
            #         hx_get="/export",
            #         cls=f"text-green-300 hover:text-green-100 p-2 w-full border-green-300 border-2 hover:border-green-100 {'hidden' if not gen_containers else ''}",
            #     ),
            #     cls="flex justify-center gap-4 w-2/3",
            # ),
            fh.Div(*gen_containers[::-1], id="gen-list", cls="flex flex-col justify-center items-center gap-4"),
            cls="flex flex-col justify-center items-center gap-4",
        )

    def developer_page(session):
        key_containers = [key_request_preview(key, session) for key in session.get("granted_keys", [])]
        return fh.Main(
            fh.Group(
                fh.Button(
                    "Request New Key",
                    id="request-new-key",
                    hx_post="/request-key",
                    target_id="api-key-table",
                    hx_swap="afterbegin",
                    cls="text-blue-300 hover:text-blue-100 p-2 w-full md:w-1/3 border-blue-300 border-2 hover:border-blue-100",
                ),
                fh.Button(
                    "Clear all",
                    id="clear-keys",
                    hx_post="/clear-keys",
                    target_id="api-key-table",
                    hx_swap="innerHTML",
                    cls=f"text-red-300 hover:text-red-100 p-2 w-full md:w-1/3 border-red-300 border-2 hover:border-red-100 {'hidden' if not key_containers else ''}",
                ),
                cls="flex flex-col md:flex-row justify-center gap-4 w-2/3",
            ),
            fh.Table(
                fh.Tr(
                    fh.Th("Key"),
                    fh.Th("Granted At"),
                    cls="font-bold",
                ),
                fh.Tbody(*key_containers[::-1], id="api-key-table"),
                cls="text-sm md:text-lg w-2/3 border-slate-500 border-2",
            ),
            cls="flex flex-col justify-center items-center gap-4",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            get_balance(),  # Live-updating balance
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

    # Helper fns

    ## generate the response (in a separate thread)
    @fh.threaded
    def generate_and_save(g) -> None:
        response = requests.post(f"{os.getenv('API_URL')}/api-key")
        if not response.ok:
            g.response = "Failed with status code: " + str(response.status_code)
        else:
            api_key = response.json()
            response = requests.post(
                os.getenv("API_URL"), json={"image_url": g.image_url}, headers={"X-API-Key": api_key}
            )
            if not response.ok:
                g.response = "Failed with status code: " + str(response.status_code)
            else:
                g.response = response.json()
        # TODO: uncomment for debugging
        # g.response = "temp"
        gens.update(g)

    # Routes
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

    ## developer page
    @f_app.get("/developer")
    def developer(session):
        return fh.Title(NAME + " | " + "developer"), fh.Div(
            nav(),
            developer_page(session),
            toast_container(),
            footer(),
            cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
        )

    ## api key request
    @f_app.post("/request-key")
    def generate_key(session):
        # Check for session ID
        if "session_id" not in session:
            fh.add_toast(session, "Please refresh the page", "error")
            return None

        # Clear button
        clear_button = (
            fh.Button(
                "Clear all",
                id="clear-keys",
                hx_post="/clear-keys",
                target_id="api-key-table",
                hx_swap="innerHTML",
                hx_swap_oob="true",
                cls="text-red-300 hover:text-red-100 p-2 w-full md:w-1/3 border-red-300 border-2 hover:border-red-100",
            ),
        )

        # Request a new key
        response = requests.post(f"{os.getenv('API_URL')}/api-key")
        if not response.ok:
            fh.add_toast(session, "Failed to request key", "error")
            return None
        else:
            api_key = response.json()
            curr_time = time.time()
            curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_time))
            key = {"key": api_key, "granted_at": curr_time}
            session.setdefault("granted_keys", []).append(key)
            return key_request_preview(key, session), clear_button

    ## pending preview keeps polling this route until we return the image preview
    @f_app.get("/gens/{id}")
    def preview(id: int, session):
        return generation_preview(gens.get(id), session)

    ## likewise we poll to keep the key request updated
    @f_app.get("/keys/{key}")
    def key_request(key: str, session):
        for key in session.get("granted_keys", []):
            if key["key"] == key:
                return key_request_preview(key, session)

    ## Likewise we poll to keep the balance updated
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
            hx_trigger="every 2s",
            hx_swap="outerHTML",
            cls="flex flex-col gap-0.5",
        )

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
    def generate_text(session, image_url: str = None):
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

        # Warn if we're out of balance
        curr_balance = get_curr_balance()
        if curr_balance.balance < 1:
            fh.add_toast(session, "Out of balance!", "error")
            return None

        # Decrement balance
        curr_balance.balance -= 1
        global_balance.update(curr_balance)

        # Clear input and button
        clear_input = fh.Input(
            id="new-image-url", name="image-url", placeholder="Enter an image url", hx_swap_oob="true"
        )
        clear_c_button = (
            fh.Button(
                "Clear all",
                id="clear-gens",
                hx_post="/clear-gens",
                target_id="gen-list",
                hx_swap="innerHTML",
                hx_swap_oob="true",
                cls="text-red-300 hover:text-red-100 p-2 w-1/3 border-red-300 border-2 hover:border-red-100",
            ),
        )
        # clear_e_button = (
        #     fh.Button(
        #         "Export to CSV",
        #         id="export-csv",
        #         hx_get="/export",
        #         cls="text-green-300 hover:text-green-100 p-2 w-1/3 border-green-300 border-2 hover:border-green-100",
        #     ),
        # )

        # Generate as before
        g = gens.insert(Generation(image_url=image_url, session_id=session["session_id"]))
        generate_and_save(g)

        return generation_preview(g, session), clear_input, clear_c_button  # , clear_e_button

    ## clear gens
    @f_app.post("/clear-gens")
    def clear_all(session):
        ids = [g.id for g in gens(where=f"session_id == '{session['session_id']}'")]
        for id in ids:
            gens.delete(id)
        clear_button = (
            fh.Button(
                "Clear all",
                id="clear-gens",
                hx_post="/clear-gens",
                target_id="gen-list",
                hx_swap="innerHTML",
                hx_swap_oob="true",
                cls="text-red-300 hover:text-red-100 p-2 w-1/3 border-red-300 border-2 hover:border-red-100 hidden",
            ),
        )
        return None, clear_button

    ## clear keys
    @f_app.post("/clear-keys")
    def clear_keys(session):
        session["granted_keys"] = []
        clear_button = (
            fh.Button(
                "Clear all",
                id="clear-keys",
                hx_post="/clear-keys",
                target_id="api-key-table",
                hx_swap="innerHTML",
                hx_swap_oob="true",
                cls="text-red-300 hover:text-red-100 p-2 w-1/3 border-red-300 border-2 hover:border-red-100 hidden",
            ),
        )
        return None, clear_button

    # ## export to CSV
    # @f_app.get("/export")
    # def export(session):
    #     output = io.StringIO()
    #     writer = csv.writer(output)
    #     writer.writerow(["image_url", "response"])
    #     for g in gens(where=f"session_id == '{session['session_id']}'"):
    #         writer.writerow([g.image_url, g.response])

    #     return fh.FileResponse(output.getvalue(), headers={"Content-Disposition": "attachment; filename=export.csv"})

    ## We send the user here to buy credits
    @f_app.get("/buy_global")
    def buy_credits(session):
        if "session_id" not in session:
            return "Error no session id"

        # Create Stripe Checkout Session
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

        # Send the USER to STRIPE
        return fh.RedirectResponse(s["url"])

    ## STRIPE sends the USER here after a payment was canceled.
    @f_app.get("/cancel")
    def cancel():
        return fh.RedirectResponse("/")

    ## STRIPE sends the USER here after a payment was 'successful'.
    @f_app.get("/success")
    def success():
        return fh.RedirectResponse("/")

    ## STRIPE calls this to tell APP when a payment was completed.
    @f_app.post("/webhook")
    async def stripe_webhook(request):
        print(request)
        print("Received webhook")
        payload = await request.body()
        payload = payload.decode("utf-8")
        signature = request.headers.get("stripe-signature")
        print(payload)

        # Verify the Stripe webhook signature
        try:
            event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
        except ValueError:
            print("Invalid payload")
            return {"error": "Invalid payload"}, 400
        except stripe.error.SignatureVerificationError:
            print("Invalid signature")
            return {"error": "Invalid signature"}, 400

        # Handle the event
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            print("Session completed", session)
            curr_balance = get_curr_balance()
            curr_balance.balance += 50
            global_balance.update(curr_balance)
            return {"status": "success"}, 200

    return f_app


# TODO:
# - add api key management
# - replace polling routes with SSE
# - add export to csv
# - add user authentication
# - add error reporting
