env ?= dev
message ?=

migrate:
	@if ! modal volume list --env=$(env) | grep -q formless-data; then \
		modal volume create --env=$(env) formless-data; \
	fi
	@if modal volume ls --env=$(env) formless-data | grep -q main.db; then \
		modal volume get --env=$(env) --force formless-data main.db db/migrations/; \
	fi
	uv run alembic -c db/migrations/alembic.ini revision --autogenerate -m "$(message)" --version-path db/migrations/versions/$(env)
	uv run alembic -c db/migrations/alembic.ini upgrade head
	modal volume put --env=$(env) --force formless-data db/migrations/main.db main.db
