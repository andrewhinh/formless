# formless

Hard handwriting understanding.

A hard handwriting image OCR system served via a [public API](https://andrewhinh--formless-api-modal-get.modal.run/), [a website](https://bit.ly/formless-fe), and [PyPI package](https://pypi.org/project/formless/) utilizing fine-tuned Qwen2-VL-7B-Instruct. Used FineWeb-inspired data quality filtering and stratified deduplication alongside SFT and DPO on worst-performing samples to reduce character error rate compared to base model.


## Usage

Use the web app:

```bash
https://bit.ly/formless-fe
```

Or hit the API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "<image-url>"}' https://andrewhinh--formless-api-modal-get.modal.run/
```

Or use the CLI:

```bash
uv run formless -i <image-url> [-v]
or
uv run formless -p <local-image-path> [-v]
```

Or use in Python:

```bash
uv add formless
```

then:

```python
from formless import scan
scan(image_url="<image-url>", verbose=1)
scan(image_path="<local-image-path>", verbose=1)
```

## Training results

Base model:

```bash
train CER: 0.9673
valid CER: 0.9606
test CER: 0.9961
```

Base quant model:

```bash
train CER: 0.9680
valid CER: 0.9622
test CER: 0.9984
```

SFT model:

```bash
train CER: 0.9771
valid CER: 0.9850
test CER: 0.9851
```

SFT quant model:

```bash
train CER: 0.9647
valid CER: 0.9611
test CER: 0.9763
```

DPO model:

```bash
train CER: 0.9772
valid CER: 0.9846
test CER: 0.9850
```

DPO quant model:

```bash
train CER: 0.9774
valid CER: 0.9849
test CER: 0.9859
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
OPENAI_API_KEY=

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
├── db                  # database.
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

Eval base model:

```bash
uv run training/eval.py --base
```

or

```bash
modal run training/eval.py --base
```

Eval quantized base model:

```bash
uv run training/eval.py --base --quant
```

or

```bash
modal run training/eval.py --base --quant
```

Run SFT:

```bash
cd training && uv sync && cd LLaMA-Factory && uv pip install -e ".[torch,metrics]" && cd .. && FORCE_TORCHRUN=1 uv run train.py --sft && cd ..
```

or

```bash
modal run training/train.py --sft
```

Eval SFT model:

```bash
uv run training/eval.py --sft
```

or

```bash
modal run training/eval.py --sft
```

Quantize the SFT model:

```bash
uv run training/quantize.py --sft
```

or

```bash
modal run training/quantize.py --sft
```

Eval quantized SFT model:

```bash
uv run training/eval.py --sft --quant
```

or

```bash
modal run training/eval.py --sft --quant
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
cd training && uv sync && cd LLaMA-Factory && uv pip install -e ".[torch,metrics]" && cd .. && FORCE_TORCHRUN=1 uv run train.py --dpo && cd ..
```

or

```bash
modal run training/train.py --dpo
```

Eval DPO model:

```bash
uv run training/eval.py --dpo
```

or

```bash
modal run training/eval.py --dpo
```

Quantize the DPO model:

```bash
uv run training/quantize.py --dpo
```

or

```bash
modal run training/quantize.py --dpo
```

Eval quantized DPO model:

```bash
uv run training/eval.py --dpo --quant
```

or

```bash
modal run training/eval.py --dpo --quant
```
