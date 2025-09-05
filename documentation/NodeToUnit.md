
# Class: Node__to_unit



URI: [file:///ines-core.yaml/NodeToUnit](file:///ines-core.yaml/NodeToUnit)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit],[Unit]<sink%200..1-%20[NodeToUnit&#124;capacity(i):float%20%3F;other_operational_cost(i):float%20%3F;investment_cost(i):float%20%3F;profile_method(i):profile_method_enum%20%3F;profile_limit_upper(i):float%20*;profile_limit_lower(i):float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]<source%200..1-%20[NodeToUnit],[Database]++-%20node__to_unit%200..*>[NodeToUnit],[Flow]^-[NodeToUnit],[Node],[Flow],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit],[Unit]<sink%200..1-%20[NodeToUnit&#124;capacity(i):float%20%3F;other_operational_cost(i):float%20%3F;investment_cost(i):float%20%3F;profile_method(i):profile_method_enum%20%3F;profile_limit_upper(i):float%20*;profile_limit_lower(i):float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]<source%200..1-%20[NodeToUnit],[Database]++-%20node__to_unit%200..*>[NodeToUnit],[Flow]^-[NodeToUnit],[Node],[Flow],[Database])

## Parents

 *  is_a: [Flow](Flow.md)

## Referenced by Class

 *  **None** *[➞node__to_unit](database__node__to_unit.md)*  <sub>0..\*</sub>  **[NodeToUnit](NodeToUnit.md)**

## Attributes


### Own

 * [➞source](nodeToUnit__source.md)  <sub>0..1</sub>
     * Range: [Node](Node.md)
 * [➞sink](nodeToUnit__sink.md)  <sub>0..1</sub>
     * Range: [Unit](Unit.md)

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
