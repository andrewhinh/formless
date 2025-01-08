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
sudo apt install libpq-dev libcairo2-dev libjpeg-dev libgif-dev openjdk-11-jdk
uv sync --all-extras --dev
uv run pre-commit install
modal setup
modal config set-environment dev
git clone git@github.com:Len-Stevens/Python-Antivirus.git frontend/Python-Antivirus
echo "alias modal='uv run modal'" >> ~/.bashrc
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE=\"~/.cache/huggingface/hub\"" >> ~/.bashrc
echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
source ~/.bashrc
```

Create a `.env` (+ `.env.dev`):

```bash
HF_TOKEN=

POSTGRES_URL=
POSTGRES_PRISMA_URL=
SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_URL=
POSTGRES_URL_NON_POOLING=
SUPABASE_JWT_SECRET=
POSTGRES_USER=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
POSTGRES_PASSWORD=
POSTGRES_DATABASE=
SUPABASE_SERVICE_ROLE_KEY=
POSTGRES_HOST=
SUPABASE_ANON_KEY=

LIVE=
DEBUG=
STRIPE_PUBLISHABLE_KEY=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
DOMAIN=
API_URL=

WANDB_API_KEY=
WANDB_ENTITY=

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
OPENAI_API_KEY=
```

### Useful Commands

Lint and format:

```bash
uv run pre-commit run --all-files
```

Migrate db (do before running the frontend/api):

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

Test the API with an example input:

```bash
modal run api/app.py
```

Serve the API locally:

```bash
uv run api/app.py
```

Serve the API on Modal:

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

Serve the web app locally:

```bash
uv run frontend/app.py
stripe listen --forward-to <url>/webhook
# update API_URL, STRIPE_WEBHOOK_SECRET, and DOMAIN in .env.dev
```

Serve the web app on Modal:

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

Label subset of data (~1000 samples) to train writing quality classifier:

```bash
modal run training/etl.py --cls
```

Run classifier training:

```bash
modal run training/train.py --cls
```

Use trained classifier to filter train/val/test data (down to ~10k samples) to train VLM using full SFT and eval:

```bash
modal run training/etl.py --sft
```

Run SFT:

```bash
modal run training/train.py --sft
```

Run trained VLM on val data and collect/manually label worst examples (~50 samples) for DPO training:

```bash
modal run training/etl.py --dpo
```

Run DPO:

```bash
modal run training/train.py --dpo
```

Quantize the DPO model:

```bash
modal run training/quantize.py
```
