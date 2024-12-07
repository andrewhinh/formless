env ?= dev
message ?= ""

migrate:
	# modal volume get --env=$(env) --force formless-data main.db db/migrations/
	uv run alembic -c db/migrations/alembic.ini revision --autogenerate -m "$(message)" --version-path db/migrations/versions/$(env)
	uv run alembic -c db/migrations/alembic.ini upgrade head
	modal volume put --env=$(env) --force formless-data db/migrations/main.db main.db
