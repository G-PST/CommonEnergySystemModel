
# Class: Storage



URI: [file:///ines-core.yaml/Storage](file:///ines-core.yaml/Storage)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20storages%200..*>[Storage&#124;storage_capacity:float%20%3F;storages_existing:float%20%3F;investment_cost:float%20%3F;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F;penalty_upward:float%20%3F;penalty_downward:float%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Storage]uses%20-.->[HasFlow],[Storage]uses%20-.->[HasPenalty],[Storage]uses%20-.->[HasInvestments],[Node]^-[Storage],[Node],[HasPenalty],[HasInvestments],[HasFlow],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20storages%200..*>[Storage&#124;storage_capacity:float%20%3F;storages_existing:float%20%3F;investment_cost:float%20%3F;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F;penalty_upward:float%20%3F;penalty_downward:float%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Storage]uses%20-.->[HasFlow],[Storage]uses%20-.->[HasPenalty],[Storage]uses%20-.->[HasInvestments],[Node]^-[Storage],[Node],[HasPenalty],[HasInvestments],[HasFlow],[Database])

## Parents

 *  is_a: [Node](Node.md)

## Uses Mixin

 *  mixin: [HasFlow](HasFlow.md)
 *  mixin: [HasPenalty](HasPenalty.md)
 *  mixin: [HasInvestments](HasInvestments.md)

## Referenced by Class

 *  **None** *[➞storages](database__storages.md)*  <sub>0..\*</sub>  **[Storage](Storage.md)**

## Attributes


### Own

 * [storage_capacity](storage_capacity.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [storages_existing](storages_existing.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [investment_cost](investment_cost.md)  <sub>0..1</sub>
     * Description: Price (in currency_year denomination) per kW
     * Range: [Float](types/Float.md)

### Inherited from Node:

 * [id](id.md)  <sub>1..1</sub>
     * Description: Unique identifier of the entity
     * Range: [Integer](types/Integer.md)
 * [name](name.md)  <sub>0..1</sub>
     * Description: Name of the entity
     * Range: [String](types/String.md)
 * [description](description.md)  <sub>0..1</sub>
     * Description: Description of the entity
     * Range: [String](types/String.md)
 * [➞node_type](node__node_type.md)  <sub>0..1</sub>
     * Description: balance, storage, commodity
     * Range: [String](types/String.md)

### Mixed in from HasFlow:

 * [➞flow_annual](hasFlow__flow_annual.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasFlow:

 * [➞flow_profile](hasFlow__flow_profile.md)  <sub>0..\*</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasFlow:

 * [➞flow_scaling_method](hasFlow__flow_scaling_method.md)  <sub>0..1</sub>
     * Description: use_profile_directly, scale_to_annual
     * Range: [flow_scaling_method_enum](flow_scaling_method_enum.md)

### Mixed in from HasPenalty:

 * [➞penalty_upward](hasPenalty__penalty_upward.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasPenalty:

 * [➞penalty_downward](hasPenalty__penalty_downward.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasInvestments:

 * [➞investment_method](hasInvestments__investment_method.md)  <sub>0..1</sub>
     * Description: not_allowed, no_limits
     * Range: [investment_method_enum](investment_method_enum.md)

### Mixed in from HasInvestments:

 * [➞interest_rate](hasInvestments__interest_rate.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasInvestments:

 * [➞lifetime](hasInvestments__lifetime.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
