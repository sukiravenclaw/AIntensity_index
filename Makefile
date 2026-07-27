PYTHON ?= python3

.PHONY: collect test clean-processed

collect:
	$(PYTHON) -m src.cli collect --config config.yaml

test:
	$(PYTHON) -m unittest discover -s tests -v

clean-processed:
	$(PYTHON) -m src.cli clean-processed --config config.yaml

