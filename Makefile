.PHONY: lint pretty

lint:
	poetry run ruff check .

pretty:
	poetry run ruff format .
