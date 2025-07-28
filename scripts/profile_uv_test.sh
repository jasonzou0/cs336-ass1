#! /bin/sh
uv run pytest --profile --profile-svg tests/test_train_bpe.py::test_train_bpe_special_tokens
