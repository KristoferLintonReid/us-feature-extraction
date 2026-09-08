# Common tasks. `make help` lists them.

IMAGE       ?= usfeat:latest
TEXLAB_DIR  ?= /path/to/TexLAB_v3
DATA        ?= sample_data
OUT         ?= out

.PHONY: help test lint payload build sample verify clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:  ## run the test suite (no weights, Octave or payload needed)
	PYTHONPATH=src python -m pytest tests -v

lint:  ## check for syntax errors and undefined names
	ruff check src tools tests --select E9,F63,F7,F82

payload:  ## encrypt the TexLab source into build/texlab.enc
	python tools/pack_texlab.py \
	  --texlab-dir $(TEXLAB_DIR) \
	  --out build/texlab.enc --key-out build/texlab.key

build:  ## build the Docker image (needs HF_TOKEN; payload optional)
	docker build -t $(IMAGE) \
	  --build-arg HF_TOKEN=$(HF_TOKEN) \
	  --build-arg TEXLAB_KEY="$$(cat build/texlab.key 2>/dev/null)" .

sample:  ## run the pipeline over the bundled sample pack
	./run.sh $(DATA) $(OUT)

verify:  ## summarise an existing output directory
	PYTHONPATH=src python -m usfeat.cli verify --out $(OUT)

clean:  ## remove outputs and caches
	rm -rf $(OUT) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
