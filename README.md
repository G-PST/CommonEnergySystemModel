[![CI](https://github.com/G-PST/CommonEnergySystemModel/actions/workflows/ci.yml/badge.svg)](https://github.com/G-PST/CommonEnergySystemModel/actions/workflows/ci.yml)
[![Docs](https://github.com/G-PST/CommonEnergySystemModel/actions/workflows/docs.yml/badge.svg)](https://g-pst.github.io/CommonEnergySystemModel/energy-system-model/schema/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

# Common Energy System Model

This repository contains a LinkML model that defines an information standard for multi-energy system modelling. The model provides a structured framework for representing energy systems with balances, storages, commodities, and other components.

It is designed to work well with other relevant standards like the [IEC-CIM](https://www.entsoe.eu/digital/common-information-model/), [qudt](https://qudt.org/). CESM has roots in in [ines-spec](https://github.com/ines-tools/ines-spec), but is built from the ground-up to conform with ontology standards and to have a clear separation between specification and implementation. It tries to consider the needs of several major modeling tools in the domain.

For users of the data transformers: read this README.md and use the generated documentation when necessary (hopefully the data just transforms)

For developers of transformers: read this README.md and then consult transformer-developer-guide.md in this same root folder.

## Overview

The Common Energy System Model (CESM) is built using [LinkML](https://linkml.io/), a powerful schema language for defining data models. CESM is at early phase and currently supports:

- Multi-energy system representation
- Nodes:
  - Balance nodes
  - Storage nodes
  - Commodity nodes
- Links between nodes
- Conversion units
- System level parameters
- Model definition

## Design principles for CESM include:

- Specification is separate from implementation (the aim is to have a common implementation as well as this would make it easier for users to transform datasets).
- Avoid domain specificity (i.e. units convert energy/matter, there is no need for the specification to separately define heat pumps and gas turbines, they just have different parametrization). This keeps the specification tractable and allows models that have been built with a layer of abstraction to avoid maintaining long compatibility lists.
- Parameters reflect physical properties (at the level applicable to energy system planning models that can manage operational detail as well as sector specific detail).
- For financial parameters, currency and inflation are not specified - each dataset should use one and the same 'currency_year' throughout.
- CESM is explicit about how something should be modeled through methods. E.g. 'piecewise_SOS2' unit_conversion_method means that conversion efficiency is presented through operating points using binary variables to enforce the ordering of those operating points.
- Methods therefore define what parameters are required and allows to check the existence of correct parameters.
- Certain model can either support or not support a specific method - easy to inform the user if something is not supported by a specific model (when transforming a dataset out to the model).
- Methods make it easier to extend the specification - usually adding a new method does not break existing functionality.
- Single definition for single thing (e.g. either efficiency or heat rate but not both). Makes life easier for the transformers interacting with INES.
- Flexibility in time: needs to allow models of different temporal scales to co-exist. Needs to distinguish between 'profile' time series and future oriented scenario-like values. Enable multi-stage multi-year modelling as well as detailed operational modelling. Support stochastic modelling.
- Non-breaking changes only within major version branches, after the version branch has been 'released'.
- Nodes do not specify which commodity they carry. Commodity association is implicit through port connections (Node_to_unit, Unit_to_node). This is intentional -- it keeps the specification flexible and domain-agnostic, and avoids requiring redundant commodity declarations on every node.

## Files Structure

### Core Model
- `model/cesm.yaml` - The main LinkML schema defining the energy system model

### Sample Data
- `data/samples/cesm-sample.yaml` - Example data demonstrating how to structure energy system information

### Processing Scripts
- `scripts/processing/` - Scripts for basic conversions (e.g. YAML to SQL) and for trying out the workflows

### Tools
- 'src/core/' - Functions to perform data conversions
- 'src/readers/' - Functions to read from files to memory
- 'src/writers/' - Functions to write from memory to files
- 'src/generated/' - Auto-generated CESM Python class from the linkml specification. Not in repository - run `gen-pydantic model/cesm.yaml > src/generated/cesm.py` to generate (needs `pip install linkml`).

## Quick Install

Prerequisites: Python 3.11+ and git.

```bash
git clone <repository-url>
cd oes-spec
pip install -e .
```

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
```

For Spine DB workflows (requires `spinedb_api`):

```bash
pip install -e ".[spine]"
```

**Generate Python classes from the schema** (required before running scripts):

```bash
pip install linkml
gen-pydantic model/cesm.yaml > src/generated/cesm_pydantic.py
```

### Verify Installation

Load the sample data into DuckDB:

```bash
python src/readers/from_yaml.py data/samples/cesm-sample.yaml artifacts/cesm.duckdb
```

See the [schema documentation](https://g-pst.github.io/CommonEnergySystemModel/energy-system-model/schema/index.html) for detailed class and enumeration reference.

## Getting Started

1. **Understanding the Model**: The `cesm.yaml` file defines the core classes:
   - `Balance`: Energy balance nodes with flow characteristics
   - `Storage`: Storage units with capacity and investment parameters
   - `Commodity`: Energy commodities with pricing
   - `Unit`: Conversion units with efficiency
   - `Link`: Transfer energy between nodes

2. **Sample Data**: The sample file shows how to structure data according to the model. It has examples for all the classes and forms a small test system.

3. **Documentation**:
   - [Schema Reference](https://g-pst.github.io/CommonEnergySystemModel/energy-system-model/schema/index.html) -- auto-generated class and enumeration documentation
   - [Transformer Developer Guide](transformer-developer-guide.md) -- architecture, DataFrame conventions, YAML syntax, and how to add new formats

## Development Environment

This project includes a devcontainer configuration that provides a consistent development environment.

### Using Dev Container in VS Code or VSCodium

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code or use DevPod with VSCodium. You will also need to have Docker Engine (and devpod for VSCodium).
2. Open this repository in VS Code or use VSCodium: cd to repository in terminal, `devpod up . --ide codium`
3. When prompted, select "Reopen in Container" or run the command (ctrl+shift+P) "Dev Containers: Reopen in Container"
4. The container will automatically build and configure the development environment with:
   - Python 3.x
   - LinkML tools
   - Required dependencies
   - Pre-configured extensions for AsciiDoc and Drawio
5. Once the repository has been reopened in the container, run `pip install -e .` to download all the python dependencies. For development, use `pip install -e ".[dev]"` to also install testing and linting tools.

### Spine Toolbox Setup

The `.spinetoolbox/project.json` file defines workflows for transforming data between CESM and various formats. Most tools (YAML to CESM, CESM to FlexTool, CESM to GridDB, etc.) use relative paths and work out of the box.

However, the FlexTool3 tool specification references an external absolute path (to a local FlexTool3 installation). If you want to run FlexTool3 within Spine Toolbox, you must update this path in `.spinetoolbox/project.json` to point to your own FlexTool3 installation's `.spinetoolbox/specifications/Tool/flextool3.json`.

FlexTool3 is optional -- the core CESM transformation tools (YAML to CESM, CESM to FlexTool format, CESM to GridDB, etc.) work without it. FlexTool3 is only needed if you want to execute the FlexTool3 solver from within the Spine Toolbox workflow.

### Generate Documentation

In order to generate the static website for local use or to be published on g-PST.github.io/CommonEnergySystemModel, two commands will need to be run

First, to generate the asciidoc files from the linkML models, run (inside devcontainer)

```BASH
python -m linkml_asciidoc_generator.main "model/cesm.yaml" -o "artifacts/documentation/modules/schema"
```

Note: `linkml-asciidoc-generator` must be installed from source (not PyPI) due to a packaging issue with template files. Install via: `pip install -e git+https://github.com/Netbeheer-Nederland/linkml-asciidoc-generator.git#egg=linkml-asciidoc-generator` or use the devcontainer which includes it.

then, to generate the HTML versions using antora, run

```BASH
antora antora-playbook.yml 
```

or if working from bash (need to have npm with appropriate packages installed):

```BASH
npx antora antora-playbook.yml
```


## Known Schema Validation Limitations

The LinkML schema defines the structure of CESM data but does not enforce all referential integrity constraints. The following validations are **not** enforced by the schema and should be checked at runtime by transformers or validation tools:

- **Port source/sink references**: `Unit_to_node` and `Node_to_unit` `source` and `sink` fields must reference existing entities (units, nodes).
- **Link node references**: `Link` `node_A` and `node_B` must reference existing nodes.
- **Group member references**: `Group_entity` members must reference existing entities.
- **PeriodFloat period references**: Period names in `PeriodFloat` values must reference existing `Period` entities.
- **Solve_pattern references**: `contains_solve_pattern` and related fields must reference valid `Solve_pattern` entities.

These are known limitations that could be addressed in future work. Until then transformer implementations should validate entity references when reading or writing CESM data (or take the risk).

## Usage

This model can be used to:
- Standardize energy system data exchange
- Generate documentation from the schema
- Create validation tools for energy system data
- Support interoperability between different energy system modeling tools

## Contributing

Contributions are welcome. Any contributions to the specification development must be carefully vetted, so best to open an issue first. As the work progresses towards data transformers between the specification and specific modelling tools, then those contributions can be more straight forward (e.g. through pull requests).

## License

This project is licensed under the MIT License - see the LICENSE file for details.