-- =============================================================================
-- SiennaGridDB Combined Schema
-- Generated from SiennaGridDB schema v0.3.0
-- Contains: schema.sql + triggers.sql + views.sql
-- =============================================================================

-- =============================================================================
-- SECTION 1: Schema (Tables and Indexes)
-- =============================================================================

-- DISCLAIMER
-- The current version of this schema only works for SQLITE >=3.45
-- When adding new functionality, think about the following:
--      1. Simplicity and ease of use over complexity,
--      2. Clear, consice and strict fields but allow for extensability,
--      3. User friendly over peformance, but consider performance always,
-- WARNING: This script should only be used while testing the schema and should not
-- be applied to existing dataset since it drops all the information it has.
DROP TABLE IF EXISTS thermal_generators;

DROP TABLE IF EXISTS renewable_generators;

DROP TABLE IF EXISTS hydro_generators;

DROP TABLE IF EXISTS storage_units;

DROP TABLE IF EXISTS prime_mover_types;

DROP TABLE IF EXISTS balancing_topologies;

DROP TABLE IF EXISTS supply_technologies;

DROP TABLE IF EXISTS storage_technology_types;

DROP TABLE IF EXISTS transmission_lines;

DROP TABLE IF EXISTS planning_regions;

DROP TABLE IF EXISTS transmission_interchanges;

DROP TABLE IF EXISTS entities;

DROP TABLE IF EXISTS time_series_associations;

DROP TABLE IF EXISTS attributes;

DROP TABLE IF EXISTS loads;

DROP TABLE IF EXISTS static_time_series;

DROP TABLE IF EXISTS entity_types;

DROP TABLE IF EXISTS supplemental_attributes;

DROP TABLE IF EXISTS arcs;

DROP TABLE IF EXISTS hydro_reservoirs;

DROP TABLE IF EXISTS hydro_reservoir_connections;

DROP TABLE IF EXISTS fuels;

DROP TABLE IF EXISTS supplemental_attributes_association;

DROP TABLE IF EXISTS transport_technologies;

PRAGMA foreign_keys = ON;

-- NOTE: This table should not be interacted directly since it gets populated
-- automatically.
-- Table of certain entities of griddb schema.
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    entity_table TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    FOREIGN KEY (entity_type) REFERENCES entity_types (name)
) strict;

-- Table of possible entity types
CREATE TABLE entity_types (
    name TEXT PRIMARY KEY,
    is_topology BOOLEAN NOT NULL DEFAULT FALSE
);

-- NOTE: Sienna-griddb follows the convention of the EIA prime mover where we
-- have a `prime_mover` and `fuel` to classify generators/storage units.
-- However, users could use any combination of `prime_mover` and `fuel` for
-- their own application. The only constraint is that the uniqueness is enforced
-- by the combination of (prime_mover, fuel)
-- Categories to classify generating units and supply technologies
CREATE TABLE prime_mover_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NULL
) strict;

CREATE TABLE fuels (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NULL
) strict;

CREATE TABLE storage_technology_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NULL
) strict;

-- Investment regions
CREATE TABLE planning_regions (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    description TEXT NULL
) strict;

-- Balancing topologies for the system. Could be either buses, or larger
-- aggregated regions.
CREATE TABLE balancing_topologies (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    area INTEGER NULL REFERENCES planning_regions (id) ON DELETE SET NULL,
    description TEXT NULL
) strict;

-- NOTE: The purpose of this table is to provide links different entities that
-- naturally have a relantionship not model dependent (e.g., transmission lines,
-- transmission interchanges, etc.).
-- Physical connection between entities.
CREATE TABLE arcs (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    CHECK (from_id <> to_id),
    FOREIGN KEY (from_id) REFERENCES entities (id) ON DELETE CASCADE,
    FOREIGN KEY (to_id) REFERENCES entities (id) ON DELETE CASCADE
) strict;

-- Existing transmission lines
CREATE TABLE transmission_lines (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    arc_id INTEGER,
    continuous_rating REAL NULL CHECK (continuous_rating >= 0),
    ste_rating REAL NULL CHECK (ste_rating >= 0),
    lte_rating REAL NULL CHECK (lte_rating >= 0),
    line_length REAL NULL CHECK (line_length >= 0),
    FOREIGN KEY (arc_id) REFERENCES arcs (id) ON DELETE CASCADE
) strict;

-- NOTE: The purpose of this table is to provide physical limits to flows
-- between areas or balancing topologies. In contrast with the transmission
-- lines, this entities are used to enforce given physical limits of certain
-- markets.
-- Transmission interchanges between two balancing topologies or areas
CREATE TABLE transmission_interchanges (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    arc_id INTEGER REFERENCES arcs(id) ON DELETE CASCADE,
    max_flow_from REAL NOT NULL,
    max_flow_to REAL NOT NULL
) strict;

-- NOTE: The purpose of these tables is to capture data of **existing units only**.
-- Table of thermal generation units (ThermalStandard, ThermalMultiStart)
CREATE TABLE thermal_generators (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    prime_mover_type TEXT NOT NULL REFERENCES prime_mover_types(name),
    fuel TEXT NOT NULL DEFAULT 'OTHER' REFERENCES fuels(name),
    balancing_topology INTEGER NOT NULL REFERENCES balancing_topologies (id) ON DELETE CASCADE,
    rating REAL NOT NULL CHECK (rating >= 0),
    base_power REAL NOT NULL CHECK (base_power > 0),
    -- Power limits (JSON: {"min": ..., "max": ...}):
    active_power_limits JSON NOT NULL,
    reactive_power_limits JSON NULL,
    -- Ramp limits (JSON: {"up": ..., "down": ...}, MW/min):
    ramp_limits JSON NULL,
    -- Time limits (JSON: {"up": ..., "down": ...}, hours):
    time_limits JSON NULL,
    -- Operational flags:
    must_run BOOLEAN NOT NULL DEFAULT FALSE,
    available BOOLEAN NOT NULL DEFAULT TRUE,
    "status" BOOLEAN NOT NULL DEFAULT FALSE,
    -- Initial setpoints:
    active_power REAL NOT NULL DEFAULT 0.0,
    reactive_power REAL NOT NULL DEFAULT 0.0,
    -- Cost (complex structure, stored as JSON):
    operation_cost JSON NOT NULL DEFAULT '{"cost_type": "THERMAL", "fixed": 0, "shut_down": 0, "start_up": 0, "variable": {"variable_cost_type": "COST", "power_units": "NATURAL_UNITS", "value_curve": {"curve_type": "INPUT_OUTPUT", "function_data": {"function_type": "LINEAR", "proportional_term": 0, "constant_term": 0}}, "vom_cost": {"curve_type": "INPUT_OUTPUT", "function_data": {"function_type": "LINEAR", "proportional_term": 0, "constant_term": 0}}}}'
);

-- Table of renewable generation units (RenewableDispatch, RenewableNonDispatch)
CREATE TABLE renewable_generators (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    prime_mover_type TEXT NOT NULL REFERENCES prime_mover_types(name),
    balancing_topology INTEGER NOT NULL REFERENCES balancing_topologies (id) ON DELETE CASCADE,
    rating REAL NOT NULL CHECK (rating >= 0),
    base_power REAL NOT NULL CHECK (base_power > 0),
    -- Renewable-specific:
    power_factor REAL NOT NULL DEFAULT 1.0 CHECK (
        power_factor > 0
        AND power_factor <= 1.0
    ),
    -- Power limits (JSON: {"min": ..., "max": ...}):
    reactive_power_limits JSON NULL,
    -- Operational flags:
    available BOOLEAN NOT NULL DEFAULT TRUE,
    -- Initial setpoints:
    active_power REAL NOT NULL DEFAULT 0.0,
    reactive_power REAL NOT NULL DEFAULT 0.0,
    -- Cost (NULL for RenewableNonDispatch):
    operation_cost JSON NULL DEFAULT '{"cost_type":"RENEWABLE","fixed":0,"variable":{"variable_cost_type":"COST","power_units":"NATURAL_UNITS","value_curve":{"curve_type":"INPUT_OUTPUT","function_data":{"function_type":"LINEAR","proportional_term":0,"constant_term":0}},"vom_cost":{"curve_type":"INPUT_OUTPUT","function_data":{"function_type":"LINEAR","proportional_term":0,"constant_term":0}}},"curtailment_cost":{"variable_cost_type":"COST","power_units":"NATURAL_UNITS","value_curve":{"curve_type":"INPUT_OUTPUT","function_data":{"function_type":"LINEAR","proportional_term":0,"constant_term":0}},"vom_cost":{"curve_type":"INPUT_OUTPUT","function_data":{"function_type":"LINEAR","proportional_term":0,"constant_term":0}}}}'
);

-- Table of hydro generation units (HydroDispatch, HydroTurbine, HydroPumpTurbine)
CREATE TABLE hydro_generators (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    prime_mover_type TEXT NOT NULL DEFAULT 'HY' REFERENCES prime_mover_types(name),
    balancing_topology INTEGER NOT NULL REFERENCES balancing_topologies (id) ON DELETE CASCADE,
    rating REAL NOT NULL CHECK (rating >= 0),
    base_power REAL NOT NULL CHECK (base_power > 0),
    -- Power limits (JSON: {"min": ..., "max": ...}):
    active_power_limits JSON NOT NULL,
    reactive_power_limits JSON NULL,
    -- Ramp limits (JSON: {"up": ..., "down": ...}, MW/min):
    ramp_limits JSON NULL,
    -- Time limits (JSON: {"up": ..., "down": ...}, hours):
    time_limits JSON NULL,
    -- Operational flags:
    available BOOLEAN NOT NULL DEFAULT TRUE,
    -- Initial setpoints:
    active_power REAL NOT NULL DEFAULT 0.0,
    reactive_power REAL NOT NULL DEFAULT 0.0,
    -- HydroTurbine/HydroPumpTurbine fields (nullable for HydroDispatch):
    powerhouse_elevation REAL NULL DEFAULT 0.0 CHECK (powerhouse_elevation >= 0),
    -- Outflow limits (JSON: {"min": ..., "max": ...}):
    outflow_limits JSON NULL,
    conversion_factor REAL NULL DEFAULT 1.0 CHECK (conversion_factor > 0),
    travel_time REAL NULL CHECK (travel_time >= 0),
    -- Cost:
    operation_cost JSON NOT NULL DEFAULT '{"cost_type": "HYDRO_GEN", "fixed": 0.0, "variable": {"variable_cost_type": "COST", "power_units": "NATURAL_UNITS", "value_curve": {"curve_type": "INPUT_OUTPUT", "function_data": {"function_type": "LINEAR", "proportional_term": 0, "constant_term": 0}}, "vom_cost": {"curve_type": "INPUT_OUTPUT", "function_data": {"function_type": "LINEAR", "proportional_term": 0, "constant_term": 0}}}}' -- Note: efficiency (varies by type), turbine_type, and HydroPumpTurbine-specific
    -- fields (active_power_limits_pump, etc.) are stored in the attributes table
);

-- NOTE: The purpose of this table is to capture data of **existing storage units only**.
-- Table of energy storage units (including PHES or other kinds),
CREATE TABLE storage_units (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    prime_mover_type TEXT NOT NULL REFERENCES prime_mover_types(name),
    storage_technology_type TEXT NOT NULL REFERENCES storage_technology_types(name),
    balancing_topology INTEGER NOT NULL REFERENCES balancing_topologies (id) ON DELETE CASCADE,
    rating REAL NOT NULL CHECK (rating >= 0),
    base_power REAL NOT NULL CHECK (base_power > 0),
    -- Storage capacity and limits (JSON: {"min": ..., "max": ...}):
    storage_capacity REAL NOT NULL CHECK (storage_capacity >= 0),
    storage_level_limits JSON NOT NULL,
    initial_storage_capacity_level REAL NOT NULL CHECK (initial_storage_capacity_level >= 0),
    -- Power limits (JSON: {"min": ..., "max": ...}, input = charging, output = discharging):
    input_active_power_limits JSON NOT NULL,
    output_active_power_limits JSON NOT NULL,
    -- Efficiency (JSON: {"in": ..., "out": ...}):
    efficiency JSON NOT NULL,
    -- Reactive power (JSON: {"min": ..., "max": ...}):
    reactive_power_limits JSON NULL,
    -- Initial setpoints:
    active_power REAL NOT NULL DEFAULT 0.0,
    reactive_power REAL NOT NULL DEFAULT 0.0,
    -- Status:
    available BOOLEAN NOT NULL DEFAULT TRUE,
    -- Storage-specific with defaults:
    conversion_factor REAL NOT NULL DEFAULT 1.0 CHECK (conversion_factor > 0),
    storage_target REAL NOT NULL DEFAULT 0.0,
    cycle_limits INTEGER NOT NULL DEFAULT 10000 CHECK (cycle_limits > 0),
    -- Cost:
    operation_cost JSON NULL
);

-- Topological hydro reservoirs
CREATE TABLE hydro_reservoirs (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    available BOOLEAN NOT NULL DEFAULT TRUE,
    -- Storage level limits (JSON: {"min": ..., "max": ...}):
    storage_level_limits JSON NOT NULL,
    initial_level REAL NOT NULL,
    -- Spillage limits (JSON: {"min": ..., "max": ...}, nullable):
    spillage_limits JSON NULL,
    inflow REAL NOT NULL DEFAULT 0.0,
    outflow REAL NOT NULL DEFAULT 0.0,
    level_targets REAL NULL,
    intake_elevation REAL NOT NULL DEFAULT 0.0,
    -- Head to volume relationship (JSON ValueCurve):
    head_to_volume_factor JSON NOT NULL,
    -- Cost (HydroReservoirCost):
    operation_cost JSON NOT NULL DEFAULT '{"cost_type": "HYDRO_RES", "level_shortage_cost": 0.0, "level_surplus_cost": 0.0, "spillage_cost": 0.0}',
    level_data_type TEXT NOT NULL DEFAULT 'USABLE_VOLUME'
);

CREATE TABLE hydro_reservoir_connections (
    source_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    sink_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    CHECK (source_id <> sink_id),
    PRIMARY KEY (source_id, sink_id)
) strict;
-- investment for expansion problems.
-- Investment technology options for expansion problems
CREATE TABLE supply_technologies (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    prime_mover_type TEXT NOT NULL REFERENCES prime_mover_types(name),
    fuel TEXT NULL REFERENCES fuels(name),
    area INTEGER NULL REFERENCES planning_regions (id) ON DELETE SET NULL,
    balancing_topology INTEGER NULL REFERENCES balancing_topologies (id) ON DELETE SET NULL,
    scenario TEXT NULL
);

CREATE UNIQUE INDEX uq_supply_tech_all
    ON supply_technologies(prime_mover_type, fuel, scenario)
    WHERE fuel IS NOT NULL AND scenario IS NOT NULL;
CREATE UNIQUE INDEX uq_supply_tech_no_fuel
    ON supply_technologies(prime_mover_type, scenario)
    WHERE fuel IS NULL AND scenario IS NOT NULL;
CREATE UNIQUE INDEX uq_supply_tech_no_scenario
    ON supply_technologies(prime_mover_type, fuel)
    WHERE fuel IS NOT NULL AND scenario IS NULL;
CREATE UNIQUE INDEX uq_supply_tech_no_fuel_no_scenario
    ON supply_technologies(prime_mover_type)
    WHERE fuel IS NULL AND scenario IS NULL;

CREATE TABLE transport_technologies (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    arc_id INTEGER NULL REFERENCES arcs(id) ON DELETE SET NULL,
    scenario TEXT NULL
);

-- NOTE: Attributes are additional parameters that can be linked to entities.
-- The main purpose of this is when there is an important field that is not
-- capture on the entity table that should exist on the model. Example of this
-- fields are variable or fixed operation and maintenance cost or any other
-- field that its representation is hard to fit into a `integer`, `real` or
-- `text`. It must not be used for operational details since most of the should
-- be included in the `operational_data` table.
CREATE TABLE attributes (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL,
    TYPE TEXT NOT NULL,
    name TEXT NOT NULL,
    value JSON NOT NULL,
    json_type TEXT generated always AS (json_type(value)) virtual,
    FOREIGN KEY (entity_id) REFERENCES entities (id) ON DELETE CASCADE,
    UNIQUE(entity_id, name)
);

-- NOTE: Supplemental are optional parameters that can be linked to entities.
-- The main purpose of this is to provide a way to save relevant information
-- but that could or could not be used for modeling. not `text`. Examples of
-- this field are geolocation (e.g., lat, long), outages, etc.)
CREATE TABLE supplemental_attributes (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    TYPE TEXT NOT NULL,
    value JSON NOT NULL,
    json_type TEXT generated always AS (json_type (value)) virtual
);

CREATE TABLE supplemental_attributes_association (
    attribute_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities (id) ON DELETE CASCADE,
    FOREIGN KEY (attribute_id) REFERENCES supplemental_attributes (id) ON DELETE CASCADE,
    PRIMARY KEY (attribute_id, entity_id)
) strict;

CREATE TABLE time_series_associations(
    id INTEGER PRIMARY KEY,
    time_series_uuid TEXT NOT NULL,
    time_series_type TEXT NOT NULL,
    initial_timestamp TEXT NOT NULL,
    resolution TEXT NOT NULL,
    horizon TEXT,
    "interval" TEXT,
    window_count INTEGER,
    length INTEGER,
    name TEXT NOT NULL,
    owner_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    owner_type TEXT NOT NULL,
    owner_category TEXT NOT NULL,
    features TEXT NOT NULL,
    scaling_factor_multiplier TEXT NULL,
    metadata_uuid TEXT NOT NULL,
    units TEXT NULL
);
CREATE UNIQUE INDEX uq_time_series_assoc_owner_type_name_res_feat ON time_series_associations (
    owner_id,
    time_series_type,
    name,
    resolution,
    features
);
CREATE INDEX idx_time_series_assoc_uuid ON time_series_associations (time_series_uuid);


CREATE TABLE loads (
    id INTEGER PRIMARY KEY REFERENCES entities (id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    balancing_topology INTEGER NOT NULL,
    base_power REAL,
    FOREIGN KEY(balancing_topology) REFERENCES balancing_topologies (id) ON DELETE CASCADE
);

CREATE TABLE static_time_series (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL,
    idx INTEGER NOT NULL,
    value REAL NOT NULL
) strict;

CREATE INDEX idx_static_time_series_uuid_idx ON static_time_series (uuid, idx);
CREATE INDEX idx_arcs_from ON arcs (from_id);
CREATE INDEX idx_arcs_to ON arcs (to_id);

-- =============================================================================
-- SECTION 2: Triggers
-- =============================================================================

CREATE TRIGGER IF NOT EXISTS check_planning_regions_entity_exists BEFORE
INSERT ON planning_regions
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'planning_regions'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table planning_regions before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_balancing_topologies_entity_exists BEFORE
INSERT ON balancing_topologies
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'balancing_topologies'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table balancing_topologies before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_arcs_entity_exists BEFORE
INSERT ON arcs
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'arcs'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table arcs before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_transmission_lines_entity_exists BEFORE
INSERT ON transmission_lines
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'transmission_lines'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table transmission_lines before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_transmission_interchanges_entity_exists BEFORE
INSERT ON transmission_interchanges
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'transmission_interchanges'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table transmission_interchanges before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_thermal_generators_entity_exists BEFORE
INSERT ON thermal_generators
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'thermal_generators'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table thermal_generators before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_renewable_generators_entity_exists BEFORE
INSERT ON renewable_generators
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'renewable_generators'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table renewable_generators before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_hydro_generators_entity_exists BEFORE
INSERT ON hydro_generators
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'hydro_generators'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table hydro_generators before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_storage_units_entity_exists BEFORE
INSERT ON storage_units
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'storage_units'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table storage_units before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_hydro_reservoirs_entity_exists BEFORE
INSERT ON hydro_reservoirs
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'hydro_reservoirs'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table hydro_reservoirs before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_supply_technologies_entity_exists BEFORE
INSERT ON supply_technologies
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'supply_technologies'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table supply_technologies before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_transport_technologies_entity_exists BEFORE
INSERT ON transport_technologies
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'transport_technologies'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table transport_technologies before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_supplemental_attributes_entity_exists BEFORE
INSERT ON supplemental_attributes
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'supplemental_attributes'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table supplemental_attributes before insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS check_loads_entity_exists BEFORE
INSERT ON loads
    WHEN NOT EXISTS (
        SELECT 1
        FROM entities
        WHERE id = NEW.id
            AND entity_table = 'loads'
    ) BEGIN
SELECT RAISE(
        ABORT,
        'Entity ID must exist in entities table with entity_table loads before insertion'
    );
END;


-- Business Logic Validation Triggers
CREATE TRIGGER enforce_arc_entity_types_insert
AFTER
INSERT ON arcs BEGIN
SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM entities
            WHERE id = NEW.from_id
        ) THEN RAISE(ABORT, 'from_id entity does not exist')
        WHEN NOT EXISTS (
            SELECT 1
            FROM entities
            WHERE id = NEW.to_id
        ) THEN RAISE(ABORT, 'to_id entity does not exist')
        WHEN (
            SELECT et.is_topology
            FROM entities e
            JOIN entity_types et ON e.entity_type = et.name
            WHERE e.id = NEW.from_id
        ) = 0 THEN RAISE(
            ABORT,
            'Invalid from_id entity type: must be a topology type (entity_types.is_topology = 1)'
        )
        WHEN (
            SELECT et.is_topology
            FROM entities e
            JOIN entity_types et ON e.entity_type = et.name
            WHERE e.id = NEW.to_id
        ) = 0 THEN RAISE(
            ABORT,
            'Invalid to_id entity type: must be a topology type (entity_types.is_topology = 1)'
        )
    END;
END;

CREATE TRIGGER enforce_arc_entity_types_update
AFTER UPDATE OF from_id, to_id ON arcs BEGIN
SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM entities
            WHERE id = NEW.from_id
        ) THEN RAISE(ABORT, 'from_id entity does not exist')
        WHEN NOT EXISTS (
            SELECT 1
            FROM entities
            WHERE id = NEW.to_id
        ) THEN RAISE(ABORT, 'to_id entity does not exist')
        WHEN (
            SELECT et.is_topology
            FROM entities e
            JOIN entity_types et ON e.entity_type = et.name
            WHERE e.id = NEW.from_id
        ) = 0 THEN RAISE(
            ABORT,
            'Invalid from_id entity type: must be a topology type (entity_types.is_topology = 1)'
        )
        WHEN (
            SELECT et.is_topology
            FROM entities e
            JOIN entity_types et ON e.entity_type = et.name
            WHERE e.id = NEW.to_id
        ) = 0 THEN RAISE(
            ABORT,
            'Invalid to_id entity type: must be a topology type (entity_types.is_topology = 1)'
        )
    END;
END;

-- Enforce that a turbine can have at most 1 upstream reservoir
-- (i.e., at most 1 row where sink is a turbine and source is a reservoir)
CREATE TRIGGER IF NOT EXISTS enforce_turbine_single_upstream_reservoir BEFORE
INSERT ON hydro_reservoir_connections
    WHEN (
        -- Check if sink is a turbine (hydro_generators or storage_units)
        SELECT entity_table
        FROM entities
        WHERE id = NEW.sink_id
    ) IN ('hydro_generators', 'storage_units')
    AND (
        -- Check if source is a reservoir
        SELECT entity_table
        FROM entities
        WHERE id = NEW.source_id
    ) = 'hydro_reservoirs' BEGIN
SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM hydro_reservoir_connections hrc
                JOIN entities e_source ON hrc.source_id = e_source.id
            WHERE hrc.sink_id = NEW.sink_id
                AND e_source.entity_table = 'hydro_reservoirs'
        ) THEN RAISE(
            ABORT,
            'Turbine already has an upstream reservoir. Each turbine can have at most 1 upstream reservoir.'
        )
    END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_turbine_single_upstream_reservoir_update
BEFORE UPDATE OF source_id, sink_id ON hydro_reservoir_connections
    WHEN (
        SELECT entity_table
        FROM entities
        WHERE id = NEW.sink_id
    ) IN ('hydro_generators', 'storage_units')
    AND (
        SELECT entity_table
        FROM entities
        WHERE id = NEW.source_id
    ) = 'hydro_reservoirs' BEGIN
SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM hydro_reservoir_connections hrc
                JOIN entities e_source ON hrc.source_id = e_source.id
            WHERE hrc.sink_id = NEW.sink_id
                AND e_source.entity_table = 'hydro_reservoirs'
                AND hrc.rowid != OLD.rowid
        ) THEN RAISE(
            ABORT,
            'Turbine already has an upstream reservoir. Each turbine can have at most 1 upstream reservoir.'
        )
    END;
END;

-- Enforce that a turbine can have at most 1 downstream reservoir
-- (i.e., at most 1 row where source is a turbine and sink is a reservoir)
CREATE TRIGGER IF NOT EXISTS enforce_turbine_single_downstream_reservoir BEFORE
INSERT ON hydro_reservoir_connections
    WHEN (
        -- Check if source is a turbine (hydro_generators or storage_units)
        SELECT entity_table
        FROM entities
        WHERE id = NEW.source_id
    ) IN ('hydro_generators', 'storage_units')
    AND (
        -- Check if sink is a reservoir
        SELECT entity_table
        FROM entities
        WHERE id = NEW.sink_id
    ) = 'hydro_reservoirs' BEGIN
SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM hydro_reservoir_connections hrc
                JOIN entities e_sink ON hrc.sink_id = e_sink.id
            WHERE hrc.source_id = NEW.source_id
                AND e_sink.entity_table = 'hydro_reservoirs'
        ) THEN RAISE(
            ABORT,
            'Turbine already has a downstream reservoir. Each turbine can have at most 1 downstream reservoir.'
        )
    END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_turbine_single_downstream_reservoir_update
BEFORE UPDATE OF source_id, sink_id ON hydro_reservoir_connections
    WHEN (
        SELECT entity_table
        FROM entities
        WHERE id = NEW.source_id
    ) IN ('hydro_generators', 'storage_units')
    AND (
        SELECT entity_table
        FROM entities
        WHERE id = NEW.sink_id
    ) = 'hydro_reservoirs' BEGIN
SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM hydro_reservoir_connections hrc
                JOIN entities e_sink ON hrc.sink_id = e_sink.id
            WHERE hrc.source_id = NEW.source_id
                AND e_sink.entity_table = 'hydro_reservoirs'
                AND hrc.rowid != OLD.rowid
        ) THEN RAISE(
            ABORT,
            'Turbine already has a downstream reservoir. Each turbine can have at most 1 downstream reservoir.'
        )
    END;
END;

-- Reverse cascade triggers: delete from entities when child table row is deleted
CREATE TRIGGER IF NOT EXISTS delete_planning_regions_entity
AFTER DELETE ON planning_regions
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_balancing_topologies_entity
AFTER DELETE ON balancing_topologies
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_arcs_entity
AFTER DELETE ON arcs
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_transmission_lines_entity
AFTER DELETE ON transmission_lines
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_transmission_interchanges_entity
AFTER DELETE ON transmission_interchanges
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_thermal_generators_entity
AFTER DELETE ON thermal_generators
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_renewable_generators_entity
AFTER DELETE ON renewable_generators
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_hydro_generators_entity
AFTER DELETE ON hydro_generators
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_storage_units_entity
AFTER DELETE ON storage_units
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_hydro_reservoirs_entity
AFTER DELETE ON hydro_reservoirs
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_supply_technologies_entity
AFTER DELETE ON supply_technologies
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_transport_technologies_entity
AFTER DELETE ON transport_technologies
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_supplemental_attributes_entity
AFTER DELETE ON supplemental_attributes
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS delete_loads_entity
AFTER DELETE ON loads
FOR EACH ROW
BEGIN
    DELETE FROM entities WHERE id = OLD.id;
END;

-- =============================================================================
-- SECTION 3: Views
-- =============================================================================

CREATE VIEW IF NOT EXISTS operational_data AS
SELECT e.id AS entity_id,
    e.entity_table,
    e.entity_type,
    json_extract(apl.value, '$.min') AS active_power_limit_min,
    json_extract(mr.value, '$') AS must_run,
    json_extract(tl.value, '$.up') AS uptime,
    json_extract(tl.value, '$.down') AS downtime,
    json_extract(rl.value, '$.up') AS ramp_up,
    json_extract(rl.value, '$.down') AS ramp_down,
    oc.value AS operational_cost,
    json_type(oc.value) AS operational_cost_type
FROM entities e
    LEFT JOIN attributes apl ON e.id = apl.entity_id
    AND apl.name = 'active_power_limits'
    LEFT JOIN attributes mr ON e.id = mr.entity_id
    AND mr.name = 'must_run'
    LEFT JOIN attributes tl ON e.id = tl.entity_id
    AND tl.name = 'time_limits'
    LEFT JOIN attributes rl ON e.id = rl.entity_id
    AND rl.name = 'ramp_limits'
    LEFT JOIN attributes oc ON e.id = oc.entity_id
    AND oc.name = 'operation_cost'
WHERE -- Only include entities that have at least one operational attribute
    (
        apl.entity_id IS NOT NULL
        OR mr.entity_id IS NOT NULL
        OR tl.entity_id IS NOT NULL
        OR rl.entity_id IS NOT NULL
        OR oc.entity_id IS NOT NULL
    );
