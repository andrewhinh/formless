SHELL := /bin/bash

setup:
	sudo apt install libpq-dev libcairo2-dev libjpeg-dev libgif-dev openjdk-11-jdk aria2 pigz s3fs
	uv sync --all-extras --dev
	uv run pre-commit install
	modal setup
	modal config set-environment dev
	git clone git@github.com:Len-Stevens/Python-Antivirus.git
	git clone --depth 1 training/https://github.com/hiyouga/LLaMA-Factory.git
	aws configure
	echo "alias modal='uv run modal'" >> ~/.bashrc
	echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
	echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
	echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
	source ~/.bashrc


# migrate database
# Usage: make migrate ENV=<env> [MSG=<message>]
ENV ?= dev
MSG ?=

migrate:
	uv run alembic -x env=$(ENV) -c db/migrations/alembic.ini stamp head
	uv run alembic -x env=$(ENV) -c db/migrations/alembic.ini revision --autogenerate -m "$(MSG)" --version-path db/migrations/versions/$(ENV)
	uv run alembic -x env=$(ENV) -c db/migrations/alembic.ini upgrade head

data:
	mkdir training/artifacts/mathwriting-2024-excerpt
	aria2c -x 16 -s 16 -j 1 -d training/artifacts -o mathwriting-2024-excerpt.tgz https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz
	tar --use-compress-program="pigz -p 8" -xvf training/artifacts/mathwriting-2024-excerpt.tgz -C training/artifacts
	rm training/artifacts/mathwriting-2024-excerpt.tgz

upload:
	modal volume create formless-data
	modal volume put formless-data training/artifacts/mathwriting-2024-excerpt