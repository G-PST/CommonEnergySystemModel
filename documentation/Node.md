
# Class: Node



URI: [file:///ines-core.yaml/Node](file:///ines-core.yaml/Node)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Storage],[NodeToUnit]-%20source%200..1>[Node&#124;node_type:string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Link]-%20node_A%200..1>[Node],[Link]-%20node_B%200..1>[Node],[UnitToNode]-%20sink%200..1>[Node],[Node]^-[Storage],[Node]^-[Commodity],[Node]^-[Balance],[Entity]^-[Node],[UnitToNode],[NodeToUnit],[Link],[Entity],[Commodity],[Balance])](https://yuml.me/diagram/nofunky;dir:TB/class/[Storage],[NodeToUnit]-%20source%200..1>[Node&#124;node_type:string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Link]-%20node_A%200..1>[Node],[Link]-%20node_B%200..1>[Node],[UnitToNode]-%20sink%200..1>[Node],[Node]^-[Storage],[Node]^-[Commodity],[Node]^-[Balance],[Entity]^-[Node],[UnitToNode],[NodeToUnit],[Link],[Entity],[Commodity],[Balance])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Children

 * [Balance](Balance.md)
 * [Commodity](Commodity.md)
 * [Storage](Storage.md)

## Referenced by Class

 *  **None** *[➞source](nodeToUnit__source.md)*  <sub>0..1</sub>  **[Node](Node.md)**
 *  **None** *[node_A](node_A.md)*  <sub>0..1</sub>  **[Node](Node.md)**
 *  **None** *[node_B](node_B.md)*  <sub>0..1</sub>  **[Node](Node.md)**
 *  **None** *[➞sink](unitToNode__sink.md)*  <sub>0..1</sub>  **[Node](Node.md)**

## Attributes


### Own

 * [➞node_type](node__node_type.md)  <sub>0..1</sub>
     * Description: balance, storage, commodity
     * Range: [String](types/String.md)

### Inherited from Entity:

 * [id](id.md)  <sub>1..1</sub>
     * Description: Unique identifier of the entity
     * Range: [Integer](types/Integer.md)
 * [name](name.md)  <sub>0..1</sub>
     * Description: Name of the entity
     * Range: [String](types/String.md)
 * [description](description.md)  <sub>0..1</sub>
     * Description: Description of the entity
     * Range: [String](types/String.md)
