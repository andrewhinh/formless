SHELL := /bin/bash


env ?= dev
message ?=

migrate:
	uv run alembic -x env=$(env) -c db/migrations/alembic.ini stamp head
	uv run alembic -x env=$(env) -c db/migrations/alembic.ini revision --autogenerate -m "$(message)" --version-path db/migrations/versions/$(env)
	uv run alembic -x env=$(env) -c db/migrations/alembic.ini upgrade head
