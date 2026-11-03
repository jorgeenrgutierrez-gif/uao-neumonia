# Atajos reproducibles. Todo pasa por uv (prohibido pip/requirements.txt).
# Multiplataforma: funciona en macOS, Linux y Windows (cmd, PowerShell o Git Bash),
# porque cada receta solo invoca `uv`; no usa rm, grep, awk ni cp.
.PHONY: help install model run test lint format check clean

MODEL := models/conv_MLP_84.h5

help:  ## Lista los comandos disponibles
	@echo Comandos disponibles:
	@echo   make install  - Crea .venv con las versiones exactas de uv.lock
	@echo   make model    - Descarga el modelo .h5 desde el Release de GitHub
	@echo   make run      - Abre la interfaz grafica, descarga el modelo si falta
	@echo   make test     - Ejecuta las pruebas con pytest
	@echo   make lint     - Revisa estilo y errores con Ruff
	@echo   make format   - Formatea el codigo con Ruff
	@echo   make check    - Lint, formato y pruebas, lo que exige la rubrica
	@echo   make clean    - Borra caches, historial.csv y los ReporteN.pdf

install:  ## Crea .venv con las versiones exactas de uv.lock
	uv sync --frozen

model: $(MODEL)  ## Descarga el modelo si no está en models/

$(MODEL):
	uv run python scripts/download_model.py

run: $(MODEL)  ## Abre la interfaz gráfica (descarga el modelo la primera vez)
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

clean:  ## Borra cachés y artefactos generados (con Python para que sirva en Windows)
	uv run python -c "import pathlib, shutil; here = pathlib.Path('.'); dirs = [*here.rglob('__pycache__'), here / '.pytest_cache', here / '.ruff_cache']; files = [*here.glob('Reporte*.pdf'), here / 'historial.csv']; [shutil.rmtree(d, ignore_errors=True) for d in dirs if '.venv' not in d.parts]; [f.unlink() for f in files if f.exists()]"
