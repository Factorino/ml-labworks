# --- Task runner -------------------------------------------------------------
#
# The single entry point for every operation on the project. The git hooks run
# the same tools (see .pre-commit-config.yaml), so a local run and a commit
# cannot drift apart.
#
# `just --list` shows everything. Recipes are grouped; within a group the list
# is sorted alphabetically, so the order in this file only matters to readers.
#
# Caution: with `set shell` below, every line of an ordinary recipe runs in its
# own shell. Variables and traps do not survive between lines — a recipe that
# needs shared state must start with a shebang. Only the last comment line
# before a recipe becomes its description in `just --list`; a blank line
# detaches a comment block from the recipe.


set shell := ["bash", "-euo", "pipefail", "-c"]


# Runs when `just` is called without arguments.
_default:
    @just --list


# --- Dependencies and environment --------------------------------------------
# uv owns the virtualenv; none of these recipes should be replaced by pip.

# Install the locked dependencies; `--locked` fails rather than rewriting uv.lock.
[group('deps')]
sync:
    uv sync --locked

# Add the deep learning stack that work 7 needs (keras, tensorflow; ~600 MB).
[group('deps')]
sync-dl:
    uv sync --locked --group dl

# Re-resolve uv.lock after editing pyproject.toml.
[group('deps')]
lock:
    uv lock

# Install the git hooks declared in .pre-commit-config.yaml.
[group('deps')]
hooks:
    uv run pre-commit install --install-hooks


# --- Running the works -------------------------------------------------------

# Download every dataset into data/; those already cached are left alone.
[group('run')]
data:
    uv run python -c "from mllab.datasets import fetch_all; fetch_all()"

# Open JupyterLab in notebooks/.
[group('run')]
lab:
    uv run jupyter lab --notebook-dir=notebooks

# Execute a notebook top to bottom in place, as the final check before a merge.
# Example: `just run-lab 01`.
[group('run')]
run-lab number:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/lab{{ number }}.ipynb

# Execute every notebook in order; the honest answer to "does it still run?".
[group('run')]
run-all:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/lab*.ipynb


# --- Static analysis ---------------------------------------------------------
# Configured in .ruff.toml and .mypy.ini.

# Everything read-only: linter, formatter check and types.
[group('lint')]
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src

# Apply the automatic fixes that `lint` only reports.
[group('lint')]
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Run every hook over the whole repository.
[group('lint')]
pre-commit:
    uv run pre-commit run --all-files


# --- Maintenance -------------------------------------------------------------

# Remove tool caches and notebook checkpoints; leaves data/ and .venv/ alone.
[group('maintenance')]
clean:
    rm -rf .ruff_cache .mypy_cache .pytest_cache
    find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
    find . -path ./.venv -prune -o -name .ipynb_checkpoints -type d -print0 | xargs -0 rm -rf
    find . -path ./.venv -prune -o -name 'kernel-ipc-*' -type s -print0 | xargs -0 rm -f

# Delete every cached dataset; the next run downloads them again.
[group('maintenance')]
clean-data:
    find data -mindepth 1 ! -name .gitkeep -delete

# What CI would run, if there were any: the full read-only gate.
[group('lint')]
check: lint
