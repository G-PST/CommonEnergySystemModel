# Changelog

All notable changes to the Common Energy System Model (CESM) specification and tools are documented in this file.

## [Unreleased]

### Added
- CSV reader, writer, and export script (`src/readers/from_csv.py`, `src/writers/to_csv.py`, `scripts/processing/cesm_to_csv.py`)
- Network visualization script (`scripts/visualization/visualize_network.py`)
- Geographic coordinates (`latitude`, `longitude`) on Node and Unit classes
- Supported Domains documentation page
- Cross-sector coupling example (CHP plant and heat pump) in YAML examples
- `CONTRIBUTING.md`
- This changelog

### Fixed
- Standardized QUDT annotation naming from `qudt_unit` to `qudt.unit` across the schema

## 2026-03-09

### Fixed
- Pinned LinkML and linkml-runtime versions for gen-pydantic compatibility
- Committed generated Pydantic models (`cesm.py`, `cesm_pydantic.py`) to the repository

### Changed
- CI now runs gen-pydantic before tests
- Code reformatted to comply with Ruff

## 2026-03-06

### Added
- Comprehensive hand-written documentation (getting started, specification, user guide, developer guide) integrated with Antora
- Test suite (`tests/`) with schema validation, reader, writer, and transformer tests
- Devcontainer configuration

### Fixed
- Documentation link corrections
- CI pipeline fixes

## 2026-02-12

### Added
- FlexTool to CESM transformer (`src/transformers/irena_flextool/`)

## 2026-01-29

### Added
- GridDB to CESM and CESM to GridDB transformers
- CESM YAML to DuckDB reader
- CESM to FlexTool transformer
- Spine Toolbox workflow integration (`.spinetoolbox/` project)

## 2025-11-24

### Changed
- Updated schema to new LinkML format with slots, mixins, and annotations

## 2025-11-07

### Added
- Working GridDB export (`src/transformers/griddb/`)
- Working FlexTool export with updated specification

## 2025-10-28 – 2025-10-31

### Added
- First version of GridDB data transformer
- FlexTool export fixes and transformer consistency improvements

## 2025-10-03

### Added
- Basic data transformation pipeline from LinkML YAML sample to Spine DB in FlexTool format

## 2025-09-22 – 2025-09-27

### Added
- YAML reader and DataFrame-based data processing pipeline
- Python class generation from schema

## 2025-09-05 – 2025-09-11

### Added
- Antora documentation site with auto-generated schema pages
- README with design principles and project overview

## 2025-08-18

### Added
- Schema and sample data validation

## 2025-08-13

### Added
- QUDT unit annotations and currency/reference year metadata to the schema

## 2025-07-08

### Added
- Spine DB writer (`src/writers/to_spine_db.py`)

## 2025-06-24

### Added
- Initial commit with core specification derived from [ines-spec](https://github.com/ines-tools/ines-spec)
