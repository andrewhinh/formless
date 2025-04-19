# formless

A hard handwriting image OCR system via a public API, website, and PyPI package, utilizing a fine-tuned Qwen2.5-VL-7B-Instruct. Utilizes FineWeb-inspired data quality filtering and stratified deduplication alongside SFT and DPO on worst-performing samples to reduce character error rate by 8.18% compared to the base model.

## Usage

Use the web app:

```bash
https://bit.ly/formless-fe
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

## Training results

Base model:

```bash
train CER: 0.9216
valid CER: 0.9276
test CER: 0.9430
```

Base quant model:

```bash
train CER: 0.9192
valid CER: 0.9232
test CER: 0.9452
```

SFT model:

```bash
train CER: 0.7965
valid CER: 0.8601
test CER: 0.8659
```

SFT quant model:

```bash
train CER: 0.8380
valid CER: 1.0214
test CER: 0.8228
```

DPO model:

```bash
train CER: 0.7827
valid CER: 0.8518
test CER: 0.9757
```

DPO quant model:

```bash
train CER: 0.7979
valid CER: 1.0352
test CER: 0.9868
```

## Development

### Set Up

Set up the environment:

```bash
make setup
```

Create a `.env` (+ `.env.dev` + `.env.local`):

```bash
HF_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=
WANDB_ENTITY=

DATABASE_URL=
API_URL=
DOMAIN=

STRIPE_PUBLISHABLE_KEY=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
```

### Useful Tips

Migrate db (do before running the frontend/api):

```bash
make migrate ENV=<env> MSG=<message>
# env = main, dev, local
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
```

Serve the web app on Modal:

```bash
modal serve frontend/app.py
stripe listen --forward-to <url>/webhook
```

Deploy on dev:

```bash
modal deploy frontend/app.py
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

Optionally upload to a Modal volume:

```bash
make upload
```

Label subset of data to train writing quality classifier:

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

Use trained classifier to filter train/val/test data to train VLM using SFT:

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

Run trained VLM on train data and construct new dataset with only relabelled incorrect examples for DPO training:

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
