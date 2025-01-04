# formless

Hard handwriting understanding.

## Usage

Use the web app:

```bash
https://andrewhinh--formless-frontend-modal-get.modal.run/
```

Or hit the API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "<image-url>"}' https://andrewhinh--formless-api-modal-get.modal.run
```

Or use the CLI:

```bash
uv run formless -i <image-url> [-v]
or
uv run formless -p <local-image-path> [-v]
```

Or use in Python:

```python
from formless import scan
scan(image_url="<image-url>", verbose=1)
scan(image_path="<local-image-path>", verbose=1)
```

## Development

### Set Up

Set up the environment:

```bash
uv sync --all-extras --dev
uv run pre-commit install
export PYTHONPATH=.
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
modal setup
modal config set-environment dev
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
modal serve api/app.py
```

Deploy on dev:

```bash
modal deploy api/app.py
```

Deploy on main:

```bash
modal deploy --env=main api/app.py
```

### Frontend

Run the web app "locally":

```bash
modal serve frontend/app.py
stripe listen --forward-to <url>/webhook
# update API_URL, STRIPE_WEBHOOK_SECRET, and DOMAIN in .env.dev
```

Deploy on dev:

```bash
modal deploy frontend/app.py
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

Label subset of data (~1000 samples) to train topic & writing quality classifiers:

```bash
modal run training/etl.py --class
```

Run classifier training:

```bash
modal run training/train.py --class
```

Use trained classifiers to filter all data (down to ~10k samples) to train VLM using full SFT:

```bash
modal run training/etl.py --sft
```

Run SFT:

```bash
modal run training/train.py --sft
```

Run trained VLM on val data and collect/manually label worst examples (~50 samples):

```bash
modal run training/etl.py --dpo
```

Run DPO:

```bash
modal run training/train.py --dpo
```

Quantize the dpo model:

```bash
modal run training/quantize.py
```
