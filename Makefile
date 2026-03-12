.PHONY: install install-docs test lint format docs-schema docs-html docs gen-pydantic clean

install:
	pip install -e ".[dev]"

install-docs:
	npm install

test:
	pytest tests/

lint:
	ruff check src/ scripts/

format:
	ruff format src/ scripts/

docs-schema:
	python -m linkml_asciidoc_generator.main "model/cesm_v0.1.0.yaml" -o "artifacts/documentation/modules/schema"
	@mkdir -p artifacts/documentation/modules/schema/pages/enumeration
	@echo '= Duration\n\nDuration is an ISO 8601 duration type (xsd:duration).' > artifacts/documentation/modules/schema/pages/enumeration/Duration.adoc

docs-html:
	npx antora antora-playbook.yml

docs: docs-schema docs-html

gen-pydantic:
	gen-pydantic model/cesm_v0.1.0.yaml > src/generated/cesm_pydantic.py

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
