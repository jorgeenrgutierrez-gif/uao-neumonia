# Atajos reproducibles. Todo pasa por uv (prohibido pip/requirements.txt).
.PHONY: help install run test lint format check clean

help:  ## Lista los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Crea .venv con las versiones exactas de uv.lock
	uv sync --frozen

run:  ## Abre la interfaz gráfica
	uv run python -m src.ui

test:  ## Ejecuta las pruebas
	uv run pytest

lint:  ## Revisa estilo y errores con Ruff
	uv run ruff check .

format:  ## Formatea el código con Ruff
	uv run ruff format .

check:  ## Lint + formato + pruebas (lo que exige la rúbrica)
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

clean:  ## Borra cachés y artefactos generados
	rm -rf .pytest_cache .ruff_cache **/__pycache__ Reporte*.pdf historial.csv
