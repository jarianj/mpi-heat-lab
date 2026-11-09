CC := mpicc
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic
LDLIBS := -lm
PYTHON ?= python

SOURCES := $(wildcard src/solver/*.c)
HEADERS := $(wildcard src/solver/*.h)
BINARY := bin/heatsolve

.PHONY: build install test lint verify scale dataset train evaluate perf figures dashboard clean

build: $(BINARY)

$(BINARY): $(SOURCES) $(HEADERS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDLIBS)

install: build
	$(PYTHON) -m pip install -e ".[dev]"

test: build
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

verify: build
	$(PYTHON) -m heatlab verify

scale: build
	$(PYTHON) -m heatlab scale

dataset: build
	$(PYTHON) -m heatlab dataset

train:
	$(PYTHON) -m heatlab train

evaluate: build
	$(PYTHON) -m heatlab evaluate

perf:
	$(PYTHON) -m heatlab perf

figures:
	$(PYTHON) -m heatlab figures

dashboard: build
	$(PYTHON) -m heatlab dashboard

clean:
	rm -rf bin
