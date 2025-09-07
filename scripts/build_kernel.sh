#! /bin/sh

# 1. Build the cython package for the tokenizer

uv run python3 setup_cython.py build_ext -i

# 2. Build the jupyter python kernel from the whole project
source .venv/bin/activate
# Optional: run `uv add ipykernel --dev`
python -m ipykernel install --user --name="cs336-ass1"
exit
