
# Class: Balance



URI: [file:///ines-core.yaml/Balance](file:///ines-core.yaml/Balance)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[HasPenalty],[HasFlow],[Database]++-%20balances%200..*>[Balance&#124;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F;penalty_upward:float%20%3F;penalty_downward:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Balance]uses%20-.->[HasFlow],[Balance]uses%20-.->[HasPenalty],[Node]^-[Balance],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[HasPenalty],[HasFlow],[Database]++-%20balances%200..*>[Balance&#124;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F;penalty_upward:float%20%3F;penalty_downward:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Balance]uses%20-.->[HasFlow],[Balance]uses%20-.->[HasPenalty],[Node]^-[Balance],[Database])

## Parents

 *  is_a: [Node](Node.md)

## Uses Mixin

 *  mixin: [HasFlow](HasFlow.md)
 *  mixin: [HasPenalty](HasPenalty.md)

## Referenced by Class

 *  **None** *[➞balances](database__balances.md)*  <sub>0..\*</sub>  **[Balance](Balance.md)**

## Attributes


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
