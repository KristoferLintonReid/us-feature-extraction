# Common tasks. `make help` lists them.

IMAGE       ?= usfeat:latest
DATA        ?= sample_data
OUT         ?= out

.PHONY: help test lint build sample verify examples clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:  ## run the test suite (no model weights needed)
	PYTHONPATH=src python -m pytest tests -v

lint:  ## check for syntax errors and undefined names
	ruff check src tools tests --select E9,F63,F7,F82

build:  ## build the Docker image (needs HF_TOKEN to fetch model weights)
	docker build -t $(IMAGE) --secret id=hf_token,env=HF_TOKEN .

sample:  ## run the pipeline over the bundled sample pack
	./run.sh $(DATA) $(OUT)

examples:  ## regenerate examples/ (5-image run, figures, preview)
	PYTHONPATH=src python -m usfeat.cli extract \
	  --data sample_data --out examples/output --limit 5
	PYTHONPATH=src python tools/make_examples.py --data sample_data --out examples/figures
	PYTHONPATH=src python tools/make_preview.py --out examples/output --preview examples/preview

verify:  ## summarise an existing output directory
	PYTHONPATH=src python -m usfeat.cli verify --out $(OUT)

clean:  ## remove outputs and caches
	rm -rf $(OUT) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
