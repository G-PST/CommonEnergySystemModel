
# Class: System



URI: [file:///ines-core.yaml/System](file:///ines-core.yaml/System)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20system%200..*>[System&#124;discount_rate:float%20%3F;timeline:string%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Entity]^-[System],[Entity],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20system%200..*>[System&#124;discount_rate:float%20%3F;timeline:string%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Entity]^-[System],[Entity],[Database])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Referenced by Class

 *  **None** *[➞system](database__system.md)*  <sub>0..\*</sub>  **[System](System.md)**

## Attributes


### Own

 * [➞discount_rate](system__discount_rate.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [➞timeline](system__timeline.md)  <sub>0..\*</sub>
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
