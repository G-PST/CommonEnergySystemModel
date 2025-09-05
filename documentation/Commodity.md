
# Class: Commodity



URI: [file:///ines-core.yaml/Commodity](file:///ines-core.yaml/Commodity)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[Database]++-%20commodity%200..*>[Commodity&#124;price_per_unit:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]^-[Commodity],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[Database]++-%20commodity%200..*>[Commodity&#124;price_per_unit:float%20%3F;node_type(i):string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]^-[Commodity],[Database])

## Parents

 *  is_a: [Node](Node.md)

## Referenced by Class

 *  **None** *[➞commodity](database__commodity.md)*  <sub>0..\*</sub>  **[Commodity](Commodity.md)**

## Attributes


### Own

 * [price_per_unit](price_per_unit.md)  <sub>0..1</sub>
     * Description: Price (in currency_year denomination) per unit of the product being bought or sold
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
