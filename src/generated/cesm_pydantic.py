from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

metamodel_version = "None"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )

    @model_serializer(mode='wrap', when_used='unless-none')
    def treat_empty_lists_as_none(
            self, handler: SerializerFunctionWrapHandler,
            info: SerializationInfo) -> dict[str, Any]:
        if info.exclude_none:
            _instance = self.model_copy()
            for field, field_info in type(_instance).model_fields.items():
                if getattr(_instance, field) == [] and not(
                        field_info.is_required()):
                    setattr(_instance, field, None)
        else:
            _instance = self
        return handler(_instance, info)



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


linkml_meta = LinkMLMeta({'default_prefix': 'cesm',
     'default_range': 'float',
     'description': 'Common Energy System Model schema',
     'id': 'file:///cesm_v0.1.0.yaml',
     'imports': ['linkml:types'],
     'license': 'https://creativecommons.org/publicdomain/zero/1.0/',
     'name': 'cesm',
     'prefixes': {'cesm': {'prefix_prefix': 'cesm',
                           'prefix_reference': 'file:///cesm_v0.1.0.yaml'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'qudt': {'prefix_prefix': 'qudt',
                           'prefix_reference': 'http://qudt.org/schema/qudt/'},
                  'unit': {'prefix_prefix': 'unit',
                           'prefix_reference': 'http://qudt.org/vocab/unit/'}},
     'source_file': 'model/cesm_v0.1.0.yaml',
     'types': {'Duration': {'base': 'str',
                            'description': 'An ISO 8601 duration string (e.g. '
                                           '"PT5M30S" for 5 minutes 30 seconds). '
                                           'When loaded into Dataframes, this '
                                           'should be converted to '
                                           'pa.duration("us") or similar.',
                            'from_schema': 'file:///cesm_v0.1.0.yaml',
                            'name': 'Duration',
                            'pattern': '^-?P(\\d+Y)?(\\d+M)?(\\d+D)?(T(\\d+H)?(\\d+M)?(\\d+S)?)?$',
                            'repr': 'str',
                            'uri': 'xsd:duration'}}} )

class FlowScalingMethod(str, Enum):
    """
    How to use flow_profile and flow_annual.
    """
    use_profile_directly = "use_profile_directly"
    scale_to_annual = "scale_to_annual"


class ConversionMethod(str, Enum):
    """
    Choose how the unit converts inputs to outputs
    """
    constant_efficiency = "constant_efficiency"
    two_point_efficiency = "two_point_efficiency"


class StartupMethod(str, Enum):
    """
    Choose how the unit startup methods are treated
    """
    linear = "Online variable is continuous"
    integer = "Online variable is discrete"


class TransferMethod(str, Enum):
    """
    How to transfer between the two links.
    """
    regular_linear = "regular_linear"


class SolveMode(str, Enum):
    """
    Choice of solve process handled within the model.
    """
    single_solve = "single_solve"
    rolling_solve = "rolling_solve"


class InvestmentMethod(str, Enum):
    """
    Choice of investment method.
    """
    not_allowed = "not_allowed"
    no_limits = "no_limits"


class NodeType(str, Enum):
    """
    Limits allowed node types: Balance, Storage, Commodity
    """
    Balance = "Balance"
    Storage = "Storage"
    Commodity = "Commodity"


class CommodityType(str, Enum):
    """
    Limits allowed commodity types
    """
    fuel = "fuel"
    emission = "emission"


class GroupType(str, Enum):
    """
    Limits allowed group types
    """
    node = "node"
    """
    Defines a group of nodes that have shared constraints.
    """
    power_grid = "power_grid"
    """
    Defines a group of balance nodes that form a synchronous power grid (these nodes can also have shared constraints).
    """
    link = "link"
    """
    Defines a group of links that have shared constraints.
    """


class Equality(str, Enum):
    equal = "equal"
    """
    Left hand side must equal right hand side
    """
    greater_than = "greater_than"
    """
    Left hand side must be greater than right hand side
    """
    less_than = "less_than"
    """
    Left hand side must be less than right hand side
    """



class Entity(ConfiguredBaseModel):
    """
    Abstract top-level class that contains all other classes (except Dataset).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True, 'from_schema': 'file:///cesm_v0.1.0.yaml'})

    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Node(Entity):
    """
    Abstract class that contains all types of nodes
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True, 'from_schema': 'file:///cesm_v0.1.0.yaml'})

    node_type: Optional[NodeType] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Commodity(Node):
    """
    Nodes where the model can buy and sell commodities against an exogenous price.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    commodity_type: CommodityType = Field(default=..., description="""Type of a commodity""", json_schema_extra = { "linkml_meta": {'domain_of': ['Commodity']} })
    price_per_unit: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Price (in dataset currency using reference_year denomination) per unit of the product being bought or sold.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Commodity'],
         'is_a': 'currency_per_megawatt-hour'} })
    node_type: Optional[NodeType] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Period(Entity):
    """
    The properties of the periods available for the model.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    years_represented: Optional[float] = Field(default=None, description="""How many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple periods in one year).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Period']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Group(Entity):
    """
    Groups define constraints on multiple entities at once
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    group_type: GroupType = Field(default=..., description="""Choice of group type""", json_schema_extra = { "linkml_meta": {'domain_of': ['Group']} })
    invest_max_total: Optional[float] = Field(default=None, description="""Maximum investment to the aggregated capacity of set member flows (unit__to_node, node__to_unit, link) and storage capacity. Applies as a sum over all periods. Constant.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Group']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class GroupEntity(Entity):
    """
    Makes an entity to be a member of a group
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    group: str = Field(default=..., description="""Group name that the entity belongs to""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Group_entity', 'Dataset']} })
    entity: str = Field(default=..., description="""Entity name""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Group_entity', 'Dataset']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Constraint(Entity):
    """
    Constraints define constraints on multiple variables at once
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    constant: Optional[list[float]] = Field(default=[], description="""Right hand side of the constraint.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Constraint']} })
    sense: Optional[Equality] = Field(default=None, description="""Is the sense of the constraint equal to, greater than or less than.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Constraint']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class SolvePattern(Entity):
    """
    Defines the properties of each solve pattern.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    solve_mode: Optional[SolveMode] = Field(default=None, description="""Choice of solve process handled within the model.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern']} })
    periods_realise_operations: Optional[list[str]] = Field(default=[], description="""A list of periods from which the model will report investment results and possibly pass to the next solve.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    periods_realise_investments: Optional[list[str]] = Field(default=[], description="""A list of periods from which the model will report investment results and possibly pass to the next solve.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    periods_pass_storage_data: Optional[list[str]] = Field(default=[], description="""A list of periods from which the model will pass storage level information to the next solve or to the child solves.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    periods_additional_operations_horizon: Optional[list[str]] = Field(default=[], description="""A list of periods that will be included in dispatch optimisation but are not part of results reporting (they may be solved again in a later solve).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    periods_additional_investments_horizon: Optional[list[str]] = Field(default=[], description="""A list of periods that will be included in invest optimisation but are not part of results reporting (they may be solved again in a later solve).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    start_time_durations: Optional[list[Timeset]] = Field(default=[], description="""Contains pairs of start time and duration to define what part of the timeline is to be solved. Start times need to match a value in the timeline. Defaults to the start of the timeline and full timeline duration. Can be a list of timesets to define representative periods.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern'], 'list_elements_ordered': True} })
    rolling_jump: Optional[str] = Field(default=None, description="""How much each roll jumps forward in time.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern']} })
    rolling_additional_horizon: Optional[str] = Field(default=None, description="""How much rolling solves have additional horizon beyond the rolling_jump duration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern']} })
    time_resolution: Optional[str] = Field(default=None, description="""Time resolution the model should use. Has to be integer multiples of the time resolution of the original data.""", json_schema_extra = { "linkml_meta": {'comments': ['Variable time resolution to be added later.'],
         'domain_of': ['Solve_pattern']} })
    contains_solve_pattern: Optional[str] = Field(default=None, description="""A child solve_pattern to be executed inside of the parent solve_pattern.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Solve_pattern']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class System(Entity):
    """
    Parameters related to the whole system to be modelled.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    solve_order: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['System'], 'list_elements_ordered': True} })
    inflation_rate: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Rate of inflation for the currency_year of the dataset.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['System']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class HasFlow(ConfiguredBaseModel):
    """
    Mixin for flow (annual_flow, flow_profile) related properties.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixin': True})

    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW.h]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[Any] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethod] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasFlow']} })


class HasPenalty(ConfiguredBaseModel):
    """
    Mixin for penalty related attributes (penalty_upward, penalty_downward).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixin': True})

    penalty_upward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })


class Balance(HasPenalty, HasFlow, Node):
    """
    Nodes that maintain a balance between inputs and outputs in each time step, but do not have a storage.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixins': ['HasFlow', 'HasPenalty']})

    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW.h]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[Any] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethod] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasFlow']} })
    penalty_upward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })
    node_type: Optional[NodeType] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class HasInvestments(ConfiguredBaseModel):
    """
    Mixin for investment related attributes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixin': True})

    investment_method: Optional[InvestmentMethod] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasInvestments']} })
    discount_rate: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    payback_time: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[a]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:YR'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })


class Storage(HasInvestments, HasPenalty, HasFlow, Node):
    """
    Nodes that include a state variable to represent storage. Also maintains balance between inputs and outputs including charging and discharging of the state.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml',
         'mixins': ['HasFlow', 'HasPenalty', 'HasInvestments']})

    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    storage_capacity: Optional[float] = Field(default=None, description="""Capacity of a single storage asset.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW.h]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['Storage']} })
    storages_existing: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Number of pre-existing storage assets at the model start.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage']} })
    investment_cost: Optional[Any] = Field(default=None, description="""Overnight cost (in dataset currency using reference_year denomination) per kWh.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Storage', 'Port', 'Link'], 'is_a': 'currency_per_kilowatt-hour'} })
    fixed_cost: Optional[Any] = Field(default=None, description="""Cost of maintaining the asset (in dataset currency using reference_year denomination) per kWh per year.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt-hour_per_year'} })
    flow_annual: Optional[float] = Field(default=None, description="""Annual flow that can be used to scale the flow profile. Always positive - flow_profile defines the direction.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW.h]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW-HR'}},
         'domain_of': ['HasFlow']} })
    flow_profile: Optional[Any] = Field(default=None, description="""Flow profile that can be scaled by flow_annual. Positive values are inflow and negative values outflow from the node.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'domain_of': ['HasFlow']} })
    flow_scaling_method: Optional[FlowScalingMethod] = Field(default=None, description="""How to use flow_profile and flow_annual. Options are use_profile_directly, scale_to_annual.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasFlow']} })
    penalty_upward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Creates the commodity out of nothing using a slack variable that causes a penalty of currency_year per unit of created stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })
    penalty_downward: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Destroys the commodity into nothingness using a slack variable that causes a penalty of currency_year per unit of destroyed stuff.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasPenalty']} })
    investment_method: Optional[InvestmentMethod] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasInvestments']} })
    discount_rate: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    payback_time: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[a]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:YR'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    node_type: Optional[NodeType] = Field(default=None, description="""Limits allowed types: Balance, Storage, Commodity""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Unit(HasInvestments, Entity):
    """
    Units convert input(s) to output(s) using a ratio multiplier.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixins': ['HasInvestments']})

    efficiency: Optional[float] = Field(default=None, description="""Multiplier for turning inputs to outputs or for transferring between nodes.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Unit', 'Link']} })
    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    conversion_method: Optional[ConversionMethod] = Field(default=None, description="""Choose how the unit converts inputs to outputs""", json_schema_extra = { "linkml_meta": {'domain_of': ['Unit']} })
    startup_method: Optional[StartupMethod] = Field(default=None, description="""Choose how the unit startups are treated in the model.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Unit']} })
    units_existing: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Number of pre-existing conversion units at the model start.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Unit']} })
    startup_cost: Optional[float] = Field(default=None, description="""Cost of one full startup for one unit.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Unit']} })
    investment_method: Optional[InvestmentMethod] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasInvestments']} })
    discount_rate: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    payback_time: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[a]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:YR'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class Link(HasInvestments, Entity):
    """
    Connects two nodes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml',
         'mixins': ['HasInvestments'],
         'slot_usage': {'efficiency': {'any_of': [{'range': 'float'},
                                                  {'multivalued': True,
                                                   'range': 'float'},
                                                  {'range': 'DirectionalValue'}],
                                       'name': 'efficiency'}}})

    efficiency: Optional[Union[DirectionalValue, float]] = Field(default=None, description="""Multiplier for turning inputs to outputs or for transferring between nodes.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'},
                    {'multivalued': True, 'range': 'float'},
                    {'range': 'DirectionalValue'}],
         'domain_of': ['Unit', 'Link']} })
    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    node_A: str = Field(default=..., description="""First node of a bidirectional link""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Link']} })
    node_B: str = Field(default=..., description="""Second node of a bidirectional link""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Link']} })
    transfer_method: Optional[TransferMethod] = Field(default=None, description="""How to transfer between the two links.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Link']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of a single asset (flow or link).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    links_existing: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Number of pre-existing links at the model start""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Link']} })
    investment_cost: Optional[Any] = Field(default=None, description="""Price (in dataset currency using reference_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Storage', 'Port', 'Link'], 'is_a': 'currency_per_kilowatt'} })
    fixed_cost: Optional[Any] = Field(default=None, description="""Cost of maintaining the asset (in dataset currency using reference_year denomination) per kW per year.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt_per_year'} })
    operational_cost: Optional[Any] = Field(default=None, description="""Cost (in dataset currency using reference_year denomination) per unit of the product flowing through, usually energy in MWh.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Link'], 'is_a': 'currency_per_megawatt-hour'} })
    investment_method: Optional[InvestmentMethod] = Field(default=None, description="""Choice of investment method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['HasInvestments']} })
    discount_rate: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Discount rate of the investment.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    payback_time: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Economic payback time of the investment. Used to annualize investments.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[a]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:YR'}},
         'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['HasInvestments']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class HasProfiles(ConfiguredBaseModel):
    """
    Mixin for profile related attributes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'mixin': True})

    profile_limit_upper: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })


class Port(HasProfiles, Entity):
    """
    Ports designates an input or an output between a unit and a node.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'from_schema': 'file:///cesm_v0.1.0.yaml',
         'mixins': ['HasProfiles']})

    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    source: str = Field(default=..., description="""Name of the source entity (Unit or Node).""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    sink: str = Field(default=..., description="""Name of the sink entity (Unit or Node).""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of the port for a single unit.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Overnight (in dataset currency using reference_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt'} })
    fixed_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost of maintaining the asset (in dataset currency using reference_year denomination) per kW per year.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt_per_year'} })
    other_operational_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost (in dataset currency using reference_year denomination) per unit of the product flowing through, usually energy in MWh.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Port'],
         'is_a': 'currency_per_megawatt-hour'} })
    constraint_flow_coefficient: Optional[ConstraintFloat] = Field(default=None, description="""Multiplier between constraints and the flow. Map of constraints and respective coefficients.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class UnitToNode(Port):
    """
    An output port from a unit to a node.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml',
         'slot_usage': {'sink': {'description': 'Sink node of the unit_to_node port.',
                                 'name': 'sink',
                                 'range': 'Node'},
                        'source': {'description': 'Source unit of the unit_to_node '
                                                  'port.',
                                   'name': 'source',
                                   'range': 'Unit'}}})

    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    source: str = Field(default=..., description="""Source unit of the unit_to_node port.""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    sink: str = Field(default=..., description="""Sink node of the unit_to_node port.""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of the port for a single unit.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Overnight (in dataset currency using reference_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt'} })
    fixed_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost of maintaining the asset (in dataset currency using reference_year denomination) per kW per year.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt_per_year'} })
    other_operational_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost (in dataset currency using reference_year denomination) per unit of the product flowing through, usually energy in MWh.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Port'],
         'is_a': 'currency_per_megawatt-hour'} })
    constraint_flow_coefficient: Optional[ConstraintFloat] = Field(default=None, description="""Multiplier between constraints and the flow. Map of constraints and respective coefficients.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class NodeToUnit(Port):
    """
    An input port from a node to a unit.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml',
         'slot_usage': {'sink': {'description': 'Sink unit of the node_to_unit port.',
                                 'name': 'sink',
                                 'range': 'Unit'},
                        'source': {'description': 'Source node of the node_to_unit '
                                                  'port.',
                                   'name': 'source',
                                   'range': 'Node'}}})

    availability: Optional[float] = Field(default=None, description="""Time series for availability (to represent forced outages).""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[10*-2]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:PERCENT'}},
         'any_of': [{'range': 'float'}, {'multivalued': True, 'range': 'float'}],
         'domain_of': ['Storage', 'Unit', 'Port', 'Link']} })
    source: str = Field(default=..., description="""Source node of the node_to_unit port.""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    sink: str = Field(default=..., description="""Sink unit of the node_to_unit port.""", json_schema_extra = { "linkml_meta": {'annotations': {'is_dimension': {'tag': 'is_dimension', 'value': True}},
         'domain_of': ['Port']} })
    capacity: Optional[float] = Field(default=None, description="""Capacity of the port for a single unit.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[MW]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:MegaW'}},
         'domain_of': ['Port', 'Link']} })
    investment_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Overnight (in dataset currency using reference_year denomination) per kW.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt'} })
    fixed_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost of maintaining the asset (in dataset currency using reference_year denomination) per kW per year.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Storage', 'Port', 'Link'],
         'is_a': 'currency_per_kilowatt_per_year'} })
    other_operational_cost: Optional[Union[PeriodFloat, float]] = Field(default=None, description="""Cost (in dataset currency using reference_year denomination) per unit of the product flowing through, usually energy in MWh.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'PeriodFloat'}],
         'domain_of': ['Port'],
         'is_a': 'currency_per_megawatt-hour'} })
    constraint_flow_coefficient: Optional[ConstraintFloat] = Field(default=None, description="""Multiplier between constraints and the flow. Map of constraints and respective coefficients.""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['Port']} })
    profile_limit_upper: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    profile_limit_lower: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[1]'},
                         'qudt_unit': {'tag': 'qudt_unit', 'value': 'unit:UNITLESS'}},
         'domain_of': ['HasProfiles']} })
    name: str = Field(default=..., description="""User-facing unique name identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    semantic_id: Optional[str] = Field(default=None, description="""Optional id for semantic web integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    alternative_names: Optional[list[str]] = Field(default=[], description="""List of alternative names and aliases.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })
    description: Optional[str] = Field(default=None, description="""Description of the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Entity']} })


class DirectionalValue(ConfiguredBaseModel):
    """
    To which direction the parameter value is applied for in a link.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    forward: Optional[list[float]] = Field(default=[], description="""The parameter is applied to the forward direction (Node_A --> Node_B).""", json_schema_extra = { "linkml_meta": {'domain_of': ['DirectionalValue']} })
    reverse: Optional[list[float]] = Field(default=[], description="""The parameter is applied to the backward direction (Node_B --> Node_A).""", json_schema_extra = { "linkml_meta": {'domain_of': ['DirectionalValue']} })


class PeriodFloat(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    period: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PeriodFloat', 'Dataset']} })
    value: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PeriodFloat', 'ConstraintFloat']} })


class ConstraintFloat(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    constraint: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['ConstraintFloat', 'Dataset']} })
    value: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PeriodFloat', 'ConstraintFloat']} })


class Timeset(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml'})

    start_time: Optional[datetime ] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Timeset']} })
    duration: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Timeset']} })


class Dataset(ConfiguredBaseModel):
    """
    Dataset properties and holder for classes available in the schema.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'file:///cesm_v0.1.0.yaml', 'tree_root': True})

    currency: str = Field(default=..., description="""QUDT currency for all monetary values in the dataset""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code',
                                            'value': '[currency]'},
                         'qudt.unit': {'tag': 'qudt.unit', 'value': 'unit:CCY'}},
         'comments': ['Currency codes must exist in http://qudt.org/vocab/currency/',
                      'Validation requires external QUDT vocabulary check'],
         'domain_of': ['Dataset']} })
    reference_year: float = Field(default=..., description="""QUDT year for all monetary values in the dataset - real (as opposed to nominal) value for this year""", json_schema_extra = { "linkml_meta": {'annotations': {'qudt.ucum_code': {'tag': 'qudt.ucum_code', 'value': '[a]'},
                         'qudt.unit': {'tag': 'qudt.unit', 'value': 'unit:YR'}},
         'domain_of': ['Dataset']} })
    entity: Optional[list[str]] = Field(default=[], description="""All entities in this dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['Group_entity', 'Dataset']} })
    id: int = Field(default=..., description="""Dataset level id to distuingish between dataset versions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    timeline: Optional[list[datetime ]] = Field(default=[], description="""Time steps for which data can be entered in the dataset. Used to validate input data.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset'], 'list_elements_ordered': True} })
    balance: Optional[list[Balance]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    storage: Optional[list[Storage]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    commodity: Optional[list[Commodity]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    unit: Optional[list[Unit]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    node_to_unit: Optional[list[NodeToUnit]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    unit_to_node: Optional[list[UnitToNode]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    link: Optional[list[Link]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    group: Optional[list[Group]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Group_entity', 'Dataset']} })
    constraint: Optional[list[Constraint]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['ConstraintFloat', 'Dataset']} })
    group_entity: Optional[list[GroupEntity]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    solve_pattern: Optional[list[SolvePattern]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    period: Optional[list[Period]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PeriodFloat', 'Dataset']} })
    system: Optional[list[System]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })

    @field_validator('currency')
    def pattern_currency(cls, v):
        pattern=re.compile(r"^[A-Z]{3}$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid currency format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid currency format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('reference_year')
    def pattern_reference_year(cls, v):
        pattern=re.compile(r"^d{4}$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid reference_year format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid reference_year format: {v}"
            raise ValueError(err_msg)
        return v


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Entity.model_rebuild()
Node.model_rebuild()
Commodity.model_rebuild()
Period.model_rebuild()
Group.model_rebuild()
GroupEntity.model_rebuild()
Constraint.model_rebuild()
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
PeriodFloat.model_rebuild()
ConstraintFloat.model_rebuild()
Timeset.model_rebuild()
Dataset.model_rebuild()
