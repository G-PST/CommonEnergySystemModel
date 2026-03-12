# Contributing to CESM

Contributions are welcome. Because changes to the specification must be carefully vetted, **please open an issue first** to discuss proposed changes before submitting a pull request.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```

## Development workflow

### Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ scripts/
ruff format src/ scripts/
```

Generated files in `src/generated/` are excluded from linting.

### Running tests

```bash
pytest tests/ -v
```

### Schema changes

If you modify `model/cesm_v0.1.0.yaml`, regenerate the Python classes:

```bash
gen-pydantic model/cesm_v0.1.0.yaml > src/generated/cesm_pydantic.py
```

Validate the schema:

```bash
python scripts/validation/validate_linkml_schema.py
```

### Documentation

Hand-written documentation lives in `documentation/modules/`. Auto-generated schema documentation is produced from the LinkML schema. Build the full documentation site with:

```bash
npx antora antora-playbook.yml
```

## Pull requests

- Keep PRs focused on a single change.
- Ensure `ruff check` and `pytest` pass before submitting.
- Provide a clear description of what the PR changes and why.
- Specification changes (schema, data format) require discussion in an issue first.

## Transformer contributions

Adding support for a new modeling tool is one of the most valuable contributions. See the [Developer Guide](https://g-pst.github.io/CommonEnergySystemModel/energy-system-model/developer-guide/architecture.html) and `transformer-developer-guide.md` for the architecture and step-by-step instructions.

## Reporting issues

Use [GitHub Issues](https://github.com/G-PST/CommonEnergySystemModel/issues) to report bugs or suggest enhancements. Include:

- A clear description of the problem or idea.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.
