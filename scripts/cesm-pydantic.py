from __future__ import annotations 

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal 
from enum import Enum 
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator
)


metamodel_version = "None"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )
    pass




class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'ines_core',
     'default_range': 'string',
     'id': 'file:///cesm.yaml',
     'imports': ['linkml:types'],
     'name': 'cesm',
     'prefixes': {'ines_core': {'prefix_prefix': 'ines_core',
                                'prefix_reference': 'file:///cesm.yaml'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'unit': {'prefix_prefix': 'unit',
                           'prefix_reference': 'http://qudt.org/vocab/unit/'}},
     'source_file': 'model/cesm.yaml'} )

class FlowScalingMethodEnum(str, Enum):
    """
    How to use flow_profile and flow_annual.
    """
    use_profile_directly = "use_profile_directly"
    scale_to_annual = "scale_to_annual"


class ConversionMethodEnum(str, Enum):
    """
    Choose how the unit converts inputs to outputs
    """
    constant_efficiency = "constant_efficiency"


class TransferMethodEnum(str, Enum):
    """
    How to transfer between the two links.
    """
    regular_linear = "regular_linear"


class SolveModeEnum(str, Enum):
    """
    Choice of solve process handled within the model.
    """
    single_solve = "single_solve"


class InvestmentMethodEnum(str, Enum):
    """
    Choice of investment method.
    """
    not_allowed = "not_allowed"
    no_limits = "no_limits"


class NodeTypeEnum(str, Enum):
    """
    Limits allowed node types: Balance, Storage, Commodity
    """
    Balance = "Balance"
    Storage = "Storage"
    Commodity = "Commodity"



class Entity(ConfiguredBaseModel):
    """
    Abstract top-level class that contains all other classes (except Database).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'from_schema': 'file:///cesm.yaml',
         'unique_keys': {'internal_id': {'description': 'Internal integer ID must be '
                                                        'unique.',
                                         'unique_key_name': 'internal_id',
                                         'unique_key_slots': ['id']}}})

    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class Node(Entity):
    """
    Abstract class that contains all types of nodes
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True, 'from_schema': 'file:///cesm.yaml'})

    node_type: Optional[NodeTypeEnum] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'alias': 'node_type', 'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class Commodity(Node):
    """
    Nodes where the model can buy and sell commodities against an exogenous price.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    price_per_unit: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per unit of the product being bought or sold.""", json_schema_extra = { "linkml_meta": {'alias': 'price_per_unit', 'domain_of': ['Commodity']} })
    node_type: Optional[NodeTypeEnum] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'alias': 'node_type', 'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class SolvePattern(Entity):
    """
    Defines a sequence of solves and the properties of each solve.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    solve_mode: Optional[SolveModeEnum] = Field(default=None, description="""Choice of solve process handled within the model.""", json_schema_extra = { "linkml_meta": {'alias': 'solve_mode', 'domain_of': ['Solve_pattern']} })
    start_time: Optional[int] = Field(default=None, description="""Start time of the solve - needs to match a value in the timeline.""", json_schema_extra = { "linkml_meta": {'alias': 'start_time',
         'comments': ['Integer to keep the sample simple for now. Will be made to use '
                      'ISO 8601 format datetime.'],
         'domain_of': ['Solve_pattern']} })
    duration: Optional[int] = Field(default=None, description="""Duration of the solve in time steps.""", json_schema_extra = { "linkml_meta": {'alias': 'duration',
         'comments': ['Integer to keep the sample simple for now. Will be made to use '
                      'ISO 8601 format duration.'],
         'domain_of': ['Solve_pattern']} })
    time_resolution: Optional[int] = Field(default=None, description="""Time resolution the model should use. Multiples of the time resolution of the original data.""", json_schema_extra = { "linkml_meta": {'alias': 'time_resolution',
         'comments': ['Integer to keep the sample simple for now. Will be made to use '
                      'ISO 8601 format duration.',
                      'Variable time resolution to be added later.'],
         'domain_of': ['Solve_pattern']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class System(Entity):
    """
    Parameters related to the whole system to be modelled.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    inflation_rate: Optional[float] = Field(default=None, description="""Rate of inflation from the currency_year of the database.""", json_schema_extra = { "linkml_meta": {'alias': 'inflation_rate',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['System']} })
    timeline: Optional[list[str]] = Field(default=None, description="""Time steps for which data can be entered in the database. Used to validate input data.""", json_schema_extra = { "linkml_meta": {'alias': 'timeline', 'domain_of': ['System']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class HasFlow(ConfiguredBaseModel):
    """
    Mixin for flow (annual_flow, flow_profile) related properties.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixin': True})

    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_annual',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[list[float]] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_profile',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethodEnum] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_scaling_method', 'domain_of': ['HasFlow']} })


class HasPenalty(ConfiguredBaseModel):
    """
    Mixin for penalty related attributes (penalty_upward, penalty_downward).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixin': True})

    penalty_upward: Optional[float] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_upward', 'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[float] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_downward', 'domain_of': ['HasPenalty']} })


class Balance(HasPenalty, HasFlow, Node):
    """
    Nodes that maintain a balance between inputs and outputs in each time step, but do not have a storage.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixins': ['HasFlow', 'HasPenalty']})

    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_annual',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[list[float]] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_profile',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethodEnum] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_scaling_method', 'domain_of': ['HasFlow']} })
    penalty_upward: Optional[float] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_upward', 'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[float] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_downward', 'domain_of': ['HasPenalty']} })
    node_type: Optional[NodeTypeEnum] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'alias': 'node_type', 'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class HasInvestments(ConfiguredBaseModel):
    """
    Mixin for investment related attributes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixin': True})

    investment_method: Optional[InvestmentMethodEnum] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_method', 'domain_of': ['HasInvestments']} })
    discount_rate: Optional[float] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'alias': 'discount_rate', 'domain_of': ['HasInvestments']} })
    payback_time: Optional[float] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'alias': 'payback_time', 'domain_of': ['HasInvestments']} })


class Storage(HasInvestments, HasPenalty, HasFlow, Node):
    """
    Nodes that include a state variable to represent storage. Also maintains balance between inputs and outputs including charging and discharging of the state.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml',
         'mixins': ['HasFlow', 'HasPenalty', 'HasInvestments']})

    storage_capacity: Optional[float] = Field(default=None, description="""Capacity of a single storage asset.""", json_schema_extra = { "linkml_meta": {'alias': 'storage_capacity',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['Storage']} })
    storages_existing: Optional[float] = Field(default=None, description="""Number of pre-existing storage assets at the model start.""", json_schema_extra = { "linkml_meta": {'alias': 'storages_existing',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Storage']} })
    investment_cost: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per kWh.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_cost', 'domain_of': ['Storage', 'Port', 'Link']} })
    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_annual',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[list[float]] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_profile',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethodEnum] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'alias': 'flow_scaling_method', 'domain_of': ['HasFlow']} })
    penalty_upward: Optional[float] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_upward', 'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[float] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'alias': 'penalty_downward', 'domain_of': ['HasPenalty']} })
    investment_method: Optional[InvestmentMethodEnum] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_method', 'domain_of': ['HasInvestments']} })
    discount_rate: Optional[float] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'alias': 'discount_rate', 'domain_of': ['HasInvestments']} })
    payback_time: Optional[float] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'alias': 'payback_time', 'domain_of': ['HasInvestments']} })
    node_type: Optional[NodeTypeEnum] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'alias': 'node_type', 'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class Unit(HasInvestments, Entity):
    """
    Units convert input(s) to output(s) using a ratio multiplier.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixins': ['HasInvestments']})

    efficiency: Optional[float] = Field(default=None, description="""Multiplier for turning inputs to outputs.""", json_schema_extra = { "linkml_meta": {'alias': 'efficiency',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['Unit', 'Link']} })
    conversion_method: Optional[ConversionMethodEnum] = Field(default=None, description="""Choose how the unit converts inputs to outputs""", json_schema_extra = { "linkml_meta": {'alias': 'conversion_method', 'domain_of': ['Unit']} })
    units_existing: Optional[float] = Field(default=None, description="""Number of pre-existing conversion units at the model start.""", json_schema_extra = { "linkml_meta": {'alias': 'units_existing',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Unit']} })
    investment_method: Optional[InvestmentMethodEnum] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_method', 'domain_of': ['HasInvestments']} })
    discount_rate: Optional[float] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'alias': 'discount_rate', 'domain_of': ['HasInvestments']} })
    payback_time: Optional[float] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'alias': 'payback_time', 'domain_of': ['HasInvestments']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class Link(HasInvestments, Entity):
    """
    Connects two nodes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml',
         'mixins': ['HasInvestments'],
         'slot_usage': {'efficiency': {'any_of': [{'range': 'float'},
                                                  {'range': 'EfficiencyValue'}],
                                       'name': 'efficiency'}}})

    efficiency: Optional[Union[EfficiencyValue, float]] = Field(default=None, description="""Multiplier for turning inputs to outputs.""", json_schema_extra = { "linkml_meta": {'alias': 'efficiency',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'EfficiencyValue'}],
         'domain_of': ['Unit', 'Link']} })
    node_A: Optional[str] = Field(default=None, description="""First node of a bidirectional link""", json_schema_extra = { "linkml_meta": {'alias': 'node_A', 'domain_of': ['Link']} })
    node_B: Optional[str] = Field(default=None, description="""Second node of a bidirectional link""", json_schema_extra = { "linkml_meta": {'alias': 'node_B', 'domain_of': ['Link']} })
    transfer_method: Optional[TransferMethodEnum] = Field(default=None, description="""How to transfer between the two links.""", json_schema_extra = { "linkml_meta": {'alias': 'transfer_method', 'domain_of': ['Link']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of a single asset (flow or link).""", json_schema_extra = { "linkml_meta": {'alias': 'capacity',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    links_existing: Optional[float] = Field(default=None, description="""Number of pre-existing links at the model start""", json_schema_extra = { "linkml_meta": {'alias': 'links_existing',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Link']} })
    investment_cost: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_cost', 'domain_of': ['Storage', 'Port', 'Link']} })
    investment_method: Optional[InvestmentMethodEnum] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_method', 'domain_of': ['HasInvestments']} })
    discount_rate: Optional[float] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'alias': 'discount_rate', 'domain_of': ['HasInvestments']} })
    payback_time: Optional[float] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'alias': 'payback_time', 'domain_of': ['HasInvestments']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class HasProfiles(ConfiguredBaseModel):
    """
    Mixin for profile related attributes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'mixin': True})

    profile_limit_upper: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_upper', 'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_lower', 'domain_of': ['HasProfiles']} })


class Port(HasProfiles, Entity):
    """
    Ports designates an input or an output between a unit and a node.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'from_schema': 'file:///cesm.yaml',
         'mixins': ['HasProfiles']})

    source_name: Optional[str] = Field(default=None, description="""Name of the source entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'source_name', 'domain_of': ['Port']} })
    sink_name: Optional[str] = Field(default=None, description="""Name of the sink entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'sink_name', 'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of a single asset (flow or link).""", json_schema_extra = { "linkml_meta": {'alias': 'capacity',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_cost', 'domain_of': ['Storage', 'Port', 'Link']} })
    other_operational_cost: Optional[float] = Field(default=None, description="""Cost (in currency_year denomination) per unit of the product flowing through""", json_schema_extra = { "linkml_meta": {'alias': 'other_operational_cost', 'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_upper', 'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_lower', 'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class UnitToNode(Port):
    """
    An output port from a unit to a node.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    source: Optional[str] = Field(default=None, description="""Source unit of the unit_to_node port.""", json_schema_extra = { "linkml_meta": {'alias': 'source', 'domain_of': ['Unit_to_node', 'Node_to_unit']} })
    sink: Optional[str] = Field(default=None, description="""Sink node of the unit_to_node port.""", json_schema_extra = { "linkml_meta": {'alias': 'sink', 'domain_of': ['Unit_to_node', 'Node_to_unit']} })
    source_name: Optional[str] = Field(default=None, description="""Name of the source entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'source_name', 'domain_of': ['Port']} })
    sink_name: Optional[str] = Field(default=None, description="""Name of the sink entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'sink_name', 'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of a single asset (flow or link).""", json_schema_extra = { "linkml_meta": {'alias': 'capacity',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_cost', 'domain_of': ['Storage', 'Port', 'Link']} })
    other_operational_cost: Optional[float] = Field(default=None, description="""Cost (in currency_year denomination) per unit of the product flowing through""", json_schema_extra = { "linkml_meta": {'alias': 'other_operational_cost', 'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_upper', 'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_lower', 'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class NodeToUnit(Port):
    """
    An input port from a node to a unit.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    source: Optional[str] = Field(default=None, description="""Source node of the node_to_unit port.""", json_schema_extra = { "linkml_meta": {'alias': 'source', 'domain_of': ['Unit_to_node', 'Node_to_unit']} })
    sink: Optional[str] = Field(default=None, description="""Sink unit of the node_to_unit port.""", json_schema_extra = { "linkml_meta": {'alias': 'sink', 'domain_of': ['Unit_to_node', 'Node_to_unit']} })
    source_name: Optional[str] = Field(default=None, description="""Name of the source entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'source_name', 'domain_of': ['Port']} })
    sink_name: Optional[str] = Field(default=None, description="""Name of the sink entity (Unit of Node).""", json_schema_extra = { "linkml_meta": {'alias': 'sink_name', 'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of a single asset (flow or link).""", json_schema_extra = { "linkml_meta": {'alias': 'capacity',
         'annotations': {'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[float] = Field(default=None, description="""Price (in currency_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'alias': 'investment_cost', 'domain_of': ['Storage', 'Port', 'Link']} })
    other_operational_cost: Optional[float] = Field(default=None, description="""Cost (in currency_year denomination) per unit of the product flowing through""", json_schema_extra = { "linkml_meta": {'alias': 'other_operational_cost', 'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_upper', 'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'profile_limit_lower', 'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity']} })
    id: int = Field(default=..., description="""Auto-generated unique identifier (hidden from user, used internally for performance).""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'alias': 'semantic_id', 'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=None, description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'alias': 'alternative_names', 'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'alias': 'description', 'domain_of': ['Entity']} })


class DirectionalValue(ConfiguredBaseModel):
    """
    To which direction the parameter value is applied for in a link.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml'})

    forward: float = Field(default=..., description="""The parameter is applied to the forward direction (Node_A --> Node_B).""", json_schema_extra = { "linkml_meta": {'alias': 'forward', 'domain_of': ['DirectionalValue']} })
    reverse: float = Field(default=..., description="""The parameter is applied to the backward direction (Node_B --> Node_A).""", json_schema_extra = { "linkml_meta": {'alias': 'reverse', 'domain_of': ['DirectionalValue']} })


class EfficiencyValue(ConfiguredBaseModel):
    """
    Valid formats for the efficiency parameter.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml',
         'union_of': ['float', 'DirectionalValue']})

    pass


class Database(ConfiguredBaseModel):
    """
    Database properties and holder for classes available in the schema.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm.yaml', 'tree_root': True})

    id: int = Field(default=..., description="""Database level id to distuingish between database versions.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity', 'Database']} })
    entities: Optional[list[Entity]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'entities', 'domain_of': ['Database']} })
    balances: Optional[list[Balance]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'balances', 'domain_of': ['Database']} })
    storages: Optional[list[Storage]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'storages', 'domain_of': ['Database']} })
    commodity: Optional[list[Commodity]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'commodity', 'domain_of': ['Database']} })
    unit: Optional[list[Unit]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'unit', 'domain_of': ['Database']} })
    node_to_unit: Optional[list[NodeToUnit]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'node_to_unit', 'domain_of': ['Database']} })
    unit_to_node: Optional[list[UnitToNode]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'unit_to_node', 'domain_of': ['Database']} })
    link: Optional[list[Link]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'link', 'domain_of': ['Database']} })
    solve_pattern: Optional[list[SolvePattern]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'solve_pattern', 'domain_of': ['Database']} })
    system: Optional[list[System]] = Field(default=None, json_schema_extra = { "linkml_meta": {'alias': 'system', 'domain_of': ['Database']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Entity.model_rebuild()
Node.model_rebuild()
Commodity.model_rebuild()
SolvePattern.model_rebuild()
System.model_rebuild()
HasFlow.model_rebuild()
HasPenalty.model_rebuild()
Balance.model_rebuild()
HasInvestments.model_rebuild()
Storage.model_rebuild()
Unit.model_rebuild()
Link.model_rebuild()
HasProfiles.model_rebuild()
Port.model_rebuild()
UnitToNode.model_rebuild()
NodeToUnit.model_rebuild()
DirectionalValue.model_rebuild()
EfficiencyValue.model_rebuild()
Database.model_rebuild()

