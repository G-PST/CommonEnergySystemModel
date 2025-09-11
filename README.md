# Common Energy System Model

This repository contains a LinkML model that defines an information standard for multi-energy system modelling. The model provides a structured framework for representing energy systems with balances, storages, commodities, and other components.

It is designed to work well with other relevant standards like the [IEC-CIM](https://www.entsoe.eu/digital/common-information-model/), [qudt](https://qudt.org/)

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


## Files Structure

### Core Model
- `docs/ines-core.yaml` - The main LinkML schema defining the energy system model

### Sample Data
- `data/samples/ines-sample.yaml` - Example data demonstrating how to structure energy system information

### Processing Scripts
- `scripts/processing/` - Scripts for basic conversions (e.g. YAML to SQL) - placeholder at the moment

## Getting Started

1. **Understanding the Model**: The `ines-core.yaml` file defines the core classes:
   - `Balance`: Energy balance nodes with flow characteristics
   - `Storage`: Storage units with capacity and investment parameters
   - `Commodity`: Energy commodities with pricing
   - `Unit`: Conversion units with efficiency
   - `Link`: Transfer energy between nodes

2. **Sample Data**: The sample file shows how to structure data according to the model. It has examples for all the classes and forms a small test system.

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
5. Once the repository has been reopened in the container, run ```poetry install``` to download all the python dependencies

### Generate Documentation

In order to generate the static website that will be published on g-PST.github.io/CommonEnergySystemModel, two commands will need to be run

First, to generate the asciidoc files from the linkML models, run

```BASH

poetry run python -m linkml_asciidoc_generator.main  "model/ines-core.yaml" "artifacts/documentation/modules/schema" --test

```

then, to generate the HTML versions using antora, run

```BASH
antora antora-playmbook.yml 
```

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