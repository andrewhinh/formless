env ?= dev
message ?=

migrate:
	@if ! modal volume list --env=$(env) | grep -q formless-db; then \
		modal volume create --env=$(env) formless-db; \
	fi
	@if modal volume ls --env=$(env) formless-db | grep -q main.db; then \
		modal volume get --env=$(env) --force formless-db main.db db/migrations/; \
	fi
	uv run alembic -c db/migrations/alembic.ini revision --autogenerate -m "$(message)" --version-path db/migrations/versions/$(env)
	uv run alembic -c db/migrations/alembic.ini upgrade head
	modal volume put --env=$(env) --force formless-db db/migrations/main.db main.db
