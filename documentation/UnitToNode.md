
# Class: Unit__to_node



URI: [file:///ines-core.yaml/UnitToNode](file:///ines-core.yaml/UnitToNode)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Node]<sink%200..1-%20[UnitToNode&#124;capacity(i):float%20%3F;other_operational_cost(i):float%20%3F;investment_cost(i):float%20%3F;profile_method(i):profile_method_enum%20%3F;profile_limit_upper(i):float%20*;profile_limit_lower(i):float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Unit]<source%200..1-%20[UnitToNode],[Database]++-%20unit__to_node%200..*>[UnitToNode],[Flow]^-[UnitToNode],[Unit],[Node],[Flow],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Node]<sink%200..1-%20[UnitToNode&#124;capacity(i):float%20%3F;other_operational_cost(i):float%20%3F;investment_cost(i):float%20%3F;profile_method(i):profile_method_enum%20%3F;profile_limit_upper(i):float%20*;profile_limit_lower(i):float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Unit]<source%200..1-%20[UnitToNode],[Database]++-%20unit__to_node%200..*>[UnitToNode],[Flow]^-[UnitToNode],[Unit],[Node],[Flow],[Database])

## Parents

 *  is_a: [Flow](Flow.md)

## Referenced by Class

 *  **None** *[➞unit__to_node](database__unit__to_node.md)*  <sub>0..\*</sub>  **[UnitToNode](UnitToNode.md)**

## Attributes


### Own

 * [➞source](unitToNode__source.md)  <sub>0..1</sub>
     * Range: [Unit](Unit.md)
 * [➞sink](unitToNode__sink.md)  <sub>0..1</sub>
     * Range: [Node](Node.md)

### Inherited from Flow:

 * [id](id.md)  <sub>1..1</sub>
     * Description: Unique identifier of the entity
     * Range: [Integer](types/Integer.md)
 * [name](name.md)  <sub>0..1</sub>
     * Description: Name of the entity
     * Range: [String](types/String.md)
 * [description](description.md)  <sub>0..1</sub>
     * Description: Description of the entity
     * Range: [String](types/String.md)
 * [capacity](capacity.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [other_operational_cost](other_operational_cost.md)  <sub>0..1</sub>
     * Description: Cost (in currency_year denomination) per unit of the product flowing through
     * Range: [Float](types/Float.md)
 * [investment_cost](investment_cost.md)  <sub>0..1</sub>
     * Description: Price (in currency_year denomination) per kW
     * Range: [Float](types/Float.md)
