# formless

Handwritten + image OCR.

## Set Up

Set up the environment:

```bash
uv sync --all-extras --dev
uv run pre-commit install
modal setup
```

## Repository Structure

```bash
.
├── api                 # API.
├── frontend            # frontend.
├── src/formless        # python bindings.
├── training            # training.
```

## Development

### API

Run the API:

```bash
modal run api/app.py
```

Deploy:

```bash
modal deploy api/app.py
```

### Frontend

Run the web app:

```bash
modal serve frontend/app.py
```

Deploy:

```bash
modal deploy frontend/app.py
```

### PyPI

Run the package:

```bash
uv run formless -v
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
uv run --with formless --no-project -- formless -vv
```

### Training

Run ETL on HF dataset:

```bash
modal run ft/etl.py
```

Train the model:

```bash
modal run ft/train_modal.py
```
