.PHONY: sync format lint typecheck test check train train-baseline train-constrained

sync:
	uv sync --extra dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run ty check src scripts tests --python-version 3.12 --extra-search-path stubs \
		--ignore call-non-callable \
		--ignore invalid-argument-type \
		--ignore invalid-method-override \
		--ignore invalid-return-type \
		--ignore not-subscriptable \
		--ignore unresolved-attribute \
		--ignore unsupported-operator

test:
	uv run pytest

check: lint typecheck test

train:
	uv run python scripts/train.py --config $(CONFIG)

train-baseline:
	uv run python scripts/train.py --config configs/experiments/baseline.yaml

train-constrained:
	uv run python scripts/train.py --config configs/experiments/constrained.yaml
