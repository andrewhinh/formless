# formless

Hard handwriting understanding.

## Usage

Hit the API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "<image-url>"}' https://andrewhinh--formless-api-model-infer.modal.run
```

Or use the CLI:

```bash
uv run formless -i <image-url> [-v]
```

Soon:

- Python bindings.

## Development

### Set Up

Set up the environment:

```bash
uv sync --all-extras --dev
uv run pre-commit install
export PYTHONPATH=.
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
modal setup
```

Create a `.env` (+ `.env.dev` + `.env.local`):

```bash
API_URL=
HF_TOKEN=

LIVE=
DEBUG=
STRIPE_PUBLISHABLE_KEY=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
DOMAIN=

WANDB_API_KEY=
WANDB_ENTITY=
```

### Useful Commands

Lint and format:

```bash
uv run pre-commit run --all-files
```

Pull latest db and migrate (do before running the frontend/api):

```bash
make migrate env=<env> message=<message>
```

### Repository Structure

```bash
.
├── api                 # API.
├── frontend            # frontend.
├── src/formless        # python bindings.
├── training            # training.
```

### API

Test the API:

```bash
modal run api/app.py
```

Run the API "locally":

```bash
modal serve --env=dev api/app.py
```

Deploy on dev:

```bash
modal deploy --env=dev api/app.py
```

Deploy on main:

```bash
modal deploy --env=main api/app.py
```

### Frontend

Run the web app "locally":

```bash
modal serve --env=dev frontend/app.py
stripe listen --forward-to <url>/webhook
# update API_URL, STRIPE_WEBHOOK_SECRET, and DOMAIN in .env.dev
```

Deploy on dev:

```bash
modal deploy --env=dev frontend/app.py
# update API_URL, STRIPE_WEBHOOK_SECRET, and DOMAIN in .env.dev
```

Deploy on main:

```bash
modal deploy --env=main frontend/app.py
```

### PyPI

Run the package:

```bash
uv run formless -v
# update API_URL in src/formless/__init__.py
```

Build the package:

```bash
uvx --from build pyproject-build --installer uv
```

Upload the package:

```bash
uvx twine upload dist/*
```

Test the uploaded package:

```bash
uv run --with formless --no-project -- formless -v
```

### Training

Run ETL:

```bash
modal run training/etl.py
```
