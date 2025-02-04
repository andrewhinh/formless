# formless

Hard handwriting understanding.

## Usage

Use the web app:

```bash
https://bit.ly/formless-fe
```

Or hit the API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "<image-url>"}' https://bit.ly/formless-api
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
make setup
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

### Useful Tips

Migrate db (do before running the frontend/api):

```bash
make migrate ENV=<env> MSG=<message>
```

Visit `http://localhost:4040/` to see the Spark UI when running `training/etl.py`.

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

Download data:

```bash
make data
```

Upload to S3 (if using Modal):

```bash
make sync
```

Label subset of data (~1k samples) to train writing quality classifier:

```bash
uv run training/etl.py --cls
```

or

```bash
modal run training/etl.py --cls
```

Run classifier training:

```bash
uv run training/train.py --cls
```

or

```bash
modal run training/train.py --cls
```

Use trained classifier to filter train/val/test data (~10k samples) to train VLM using SFT:

```bash
uv run training/etl.py --sft
```

or

```bash
modal run training/etl.py --sft
```

Run SFT:

```bash
cd training && uv sync && cd LLaMA-Factory && uv pip install -e ".[torch,metrics]" && uv pip install flash-attn==2.7.2.post1 --no-build-isolation && cd .. && FORCE_TORCHRUN=1 uv run train.py --sft && cd ..
```

or

```bash
modal run training/train.py --sft
```

Quantize the SFT model:

```bash
uv pip install flash-attn==2.7.2.post1 --no-build-isolation && uv run training/quantize.py --sft
```

or

```bash
modal run training/quantize.py --sft
```

Run trained VLM on train data and construct new dataset with only relabelled incorrect examples (~1k samples) for DPO training:

```bash
uv run training/etl.py --dpo
```

or

```bash
modal run training/etl.py --dpo
```

Run DPO:

```bash
cd training && uv sync && cd LLaMA-Factory && uv pip install -e ".[torch,metrics]" && uv pip install flash-attn==2.7.2.post1 --no-build-isolation && cd .. && FORCE_TORCHRUN=1 uv run train.py --dpo && cd ..
```

or

```bash
modal run training/train.py --dpo
```

Quantize the DPO model:

````bash
uv pip install flash-attn==2.7.2.post1 --no-build-isolation && uv run training/quantize.py --dpo
``

or


```bash
modal run training/quantize.py --dpo
````
