import os
from pathlib import Path

import modal

from utils import (
    FE_ALLOW_CONCURRENT_INPUTS,
    FE_CONTAINER_IDLE_TIMEOUT,
    FE_TIMEOUT,
    IMAGE,
    NAME,
)

# Modal
IMAGE = IMAGE.pip_install(  # add Python dependencies
    "python-fasthtml==0.6.10", "simpleicons==7.21.0"
).copy_local_dir(Path(__file__).parent, "/root/")

APP_NAME = f"{NAME}-frontend"
app = modal.App(APP_NAME)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=Path(__file__).parent)],
    timeout=FE_TIMEOUT,
    container_idle_timeout=FE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=FE_ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():
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

    def main_content():
        return fh.Main(
            # fh.Div(
            #     fh.Div(
            #         fh.Label(
            #             "Upload Image:",
            #             fh.Input(
            #                 type="file",
            #                 id="file-input",
            #                 accept="image/*",
            #                 hx_post="/upload",
            #                 hx_target="#image-panel",
            #                 hx_swap="outerHTML",
            #             ),
            #             cls="flex flex-col gap-2",
            #         ),
            #         fh.Div(id="image-panel", cls="p-4 border border-gray-300"),
            #         cls="w-1/2",
            #     ),
            #     fh.Div(
            #         fh.P("OCR Results:", cls="text-lg font-bold"),
            #         fh.Div(id="ocr-results", cls="p-4 border border-gray-300"),
            #         cls="w-1/2",
            #     ),
            #     cls="flex justify-between",
            # ),
            # fh.Script(
            #     "htmx.on('htmx:afterRequest', function(evt) {"
            #     "  if (evt.detail.elt.id === 'file-input' && evt.detail.requestConfig.verb === 'post') {"
            #     "    htmx.trigger('#toast-container', 'showToast', {detail: {message: 'Upload successful'}});"
            #     "  }"
            #     "});"
            #     "htmx.on('htmx:beforeRequest', function(evt) {"
            #     "  if (evt.detail.elt.id === 'file-input' && evt.detail.requestConfig.verb === 'post') {"
            #     "    document.getElementById('ocr-results').innerHTML = '<div class=\"loading\">Processing...</div>';"
            #     "  }"
            #     "});"
            # ),
            # cls="flex flex-col gap-4 justify-center items-center flex-1",
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
    async def get():
        return fh.Title(NAME), fh.Div(
            nav(),
            main_content(),
            footer(),
            cls="flex flex-col justify-between min-h-screen bg-zinc-900 w-full",
        )

    @rt("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        parent_path = Path(__file__).parent
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    return fasthtml_app
