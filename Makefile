# Laya Decision Studio
#
# make serve     run the studio at http://127.0.0.1:$(PORT)   (default target)
# make install   install requirements into .venv (creates it if missing)
# make test      run the API tests against a stubbed Router
# make clean     remove test/build caches
# make image     build the Quay image (checkpoints baked in; slow first build)
# make push      push the image to Quay

VENV   := .venv
PYTHON := $(VENV)/bin/python

# Run configuration: make serve PORT=8000 HOST=0.0.0.0 EXTRA=--no-preload
HOST  ?= 127.0.0.1
PORT  ?= 7860
EXTRA ?=

# Image configuration: make image NAMESPACE=eformat TAG=latest
NAMESPACE ?= eformat
IMAGE     ?= quay.io/$(NAMESPACE)/laya-decision-studio
TAG       ?= latest

.PHONY: serve install test clean image push

serve: $(PYTHON)
	$(PYTHON) -m studio.app --host $(HOST) --port $(PORT) $(EXTRA)

$(PYTHON):
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -r requirements.txt

install: $(PYTHON)
	$(PYTHON) -m pip install -r requirements.txt

test: $(PYTHON)
	$(PYTHON) -m pytest studio/tests/ -q

clean:
	rm -rf .pytest_cache
	find studio -name __pycache__ -type d -prune -exec rm -rf {} +

image:
	podman build -t $(IMAGE):$(TAG) .

push: image
	podman push $(IMAGE):$(TAG)
