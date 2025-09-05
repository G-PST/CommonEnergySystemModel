
# ines-core


**metamodel version:** 1.7.0

**version:** None





### Classes

 * [Database](Database.md)
 * [DirectionalValue](DirectionalValue.md)
 * [EfficiencyValue](EfficiencyValue.md)
 * [Entity](Entity.md) - abstract high-level class
     * [Flow](Flow.md)
         * [NodeToUnit](NodeToUnit.md)
         * [UnitToNode](UnitToNode.md)
     * [Link](Link.md)
     * [Node](Node.md)
         * [Balance](Balance.md)
         * [Commodity](Commodity.md)
         * [Storage](Storage.md)
     * [SolvePattern](SolvePattern.md)
     * [System](System.md)
     * [Unit](Unit.md)

### Mixins

 * [HasFlow](HasFlow.md)
 * [HasInvestments](HasInvestments.md)
 * [HasPenalty](HasPenalty.md)
 * [HasProfiles](HasProfiles.md)

### Slots

 * [capacity](capacity.md)
 * [currency_year](currency_year.md) - QUDT currency and reference year for all monetary values in the dataset
 * [➞balances](database__balances.md)
 * [➞commodity](database__commodity.md)
 * [➞link](database__link.md)
 * [➞node__to_unit](database__node__to_unit.md)
 * [➞solve_pattern](database__solve_pattern.md)
 * [➞storages](database__storages.md)
 * [➞system](database__system.md)
 * [➞unit](database__unit.md)
 * [➞unit__to_node](database__unit__to_node.md)
 * [description](description.md) - Description of the entity
 * [efficiency](efficiency.md)
     * [Link➞efficiency](Link_efficiency.md)
 * [entity](entity.md) - All entities in this database
 * [forward](forward.md)
 * [➞flow_annual](hasFlow__flow_annual.md)
 * [➞flow_profile](hasFlow__flow_profile.md)
 * [➞flow_scaling_method](hasFlow__flow_scaling_method.md) - use_profile_directly, scale_to_annual
 * [➞interest_rate](hasInvestments__interest_rate.md)
 * [➞investment_method](hasInvestments__investment_method.md) - not_allowed, no_limits
 * [➞lifetime](hasInvestments__lifetime.md)
 * [➞penalty_downward](hasPenalty__penalty_downward.md)
 * [➞penalty_upward](hasPenalty__penalty_upward.md)
 * [➞profile_limit_lower](hasProfiles__profile_limit_lower.md)
 * [➞profile_limit_upper](hasProfiles__profile_limit_upper.md)
 * [➞profile_method](hasProfiles__profile_method.md) - upper_limit, lower_limit
 * [id](id.md) - Unique identifier of the entity
 * [investment_cost](investment_cost.md) - Price (in currency_year denomination) per kW
 * [➞transfer_method](link__transfer_method.md) - regular_linear
 * [links_existing](links_existing.md)
 * [name](name.md) - Name of the entity
 * [➞sink](nodeToUnit__sink.md)
 * [➞source](nodeToUnit__source.md)
 * [node_A](node_A.md) - First node of a bidirectional link
 * [node_B](node_B.md) - Second node of a bidirectional link
 * [➞node_type](node__node_type.md) - balance, storage, commodity
 * [other_operational_cost](other_operational_cost.md) - Cost (in currency_year denomination) per unit of the product flowing through
 * [price_per_unit](price_per_unit.md) - Price (in currency_year denomination) per unit of the product being bought or sold
 * [reverse](reverse.md)
 * [sink](sink.md) - Sink of the unidirectional flow
 * [➞duration](solvePattern__duration.md)
 * [➞solve_mode](solvePattern__solve_mode.md) - single_solve
 * [➞start_time](solvePattern__start_time.md)
 * [➞time_resolution](solvePattern__time_resolution.md)
 * [source](source.md) - Source of the unidirectional flow
 * [storage_capacity](storage_capacity.md)
 * [storages_existing](storages_existing.md)
 * [➞discount_rate](system__discount_rate.md)
 * [➞timeline](system__timeline.md)
 * [➞sink](unitToNode__sink.md)
 * [➞source](unitToNode__source.md)
 * [➞conversion_method](unit__conversion_method.md) - constant_efficiency
 * [units_existing](units_existing.md)

### Enums

 * [conversion_method_enum](conversion_method_enum.md)
 * [flow_scaling_method_enum](flow_scaling_method_enum.md)
 * [investment_method_enum](investment_method_enum.md)
 * [profile_method_enum](profile_method_enum.md)
 * [solve_model_enum](solve_model_enum.md)
 * [transfer_method_enum](transfer_method_enum.md)

### Subsets


### Types


#### Built in

 * **Bool**
 * **Curie**
 * **Decimal**
 * **ElementIdentifier**
 * **NCName**
 * **NodeIdentifier**
 * **URI**
 * **URIorCURIE**
 * **XSDDate**
 * **XSDDateTime**
 * **XSDTime**
 * **float**
 * **int**
 * **str**

#### Defined

 * [Boolean](types/Boolean.md)  (**Bool**)  - A binary (true or false) value
 * [Curie](types/Curie.md)  (**Curie**)  - a compact URI
 * [Date](types/Date.md)  (**XSDDate**)  - a date (year, month and day) in an idealized calendar
 * [DateOrDatetime](types/DateOrDatetime.md)  (**str**)  - Either a date or a datetime
 * [Datetime](types/Datetime.md)  (**XSDDateTime**)  - The combination of a date and time
 * [Decimal](types/Decimal.md)  (**Decimal**)  - A real number with arbitrary precision that conforms to the xsd:decimal specification
 * [Double](types/Double.md)  (**float**)  - A real number that conforms to the xsd:double specification
 * [Float](types/Float.md)  (**float**)  - A real number that conforms to the xsd:float specification
 * [Integer](types/Integer.md)  (**int**)  - An integer
 * [Jsonpath](types/Jsonpath.md)  (**str**)  - A string encoding a JSON Path. The value of the string MUST conform to JSON Point syntax and SHOULD dereference to zero or more valid objects within the current instance document when encoded in tree form.
 * [Jsonpointer](types/Jsonpointer.md)  (**str**)  - A string encoding a JSON Pointer. The value of the string MUST conform to JSON Point syntax and SHOULD dereference to a valid object within the current instance document when encoded in tree form.
 * [Ncname](types/Ncname.md)  (**NCName**)  - Prefix part of CURIE
 * [Nodeidentifier](types/Nodeidentifier.md)  (**NodeIdentifier**)  - A URI, CURIE or BNODE that represents a node in a model.
 * [Objectidentifier](types/Objectidentifier.md)  (**ElementIdentifier**)  - A URI or CURIE that represents an object in the model.
 * [Sparqlpath](types/Sparqlpath.md)  (**str**)  - A string encoding a SPARQL Property Path. The value of the string MUST conform to SPARQL syntax and SHOULD dereference to zero or more valid objects within the current instance document when encoded as RDF.
 * [String](types/String.md)  (**str**)  - A character string
 * [Time](types/Time.md)  (**XSDTime**)  - A time object represents a (local) time of day, independent of any particular day
 * [Uri](types/Uri.md)  (**URI**)  - a complete URI
 * [Uriorcurie](types/Uriorcurie.md)  (**URIorCURIE**)  - a URI or a CURIE
