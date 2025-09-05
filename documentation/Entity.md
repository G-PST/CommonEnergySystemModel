
# Class: Entity

abstract high-level class

URI: [file:///ines-core.yaml/Entity](file:///ines-core.yaml/Entity)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit],[System],[SolvePattern],[Node],[Link],[Flow],[Entity&#124;id:integer;name:string%20%3F;description:string%20%3F]^-[Unit],[Entity]^-[System],[Entity]^-[SolvePattern],[Entity]^-[Node],[Entity]^-[Link],[Entity]^-[Flow])](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit],[System],[SolvePattern],[Node],[Link],[Flow],[Entity&#124;id:integer;name:string%20%3F;description:string%20%3F]^-[Unit],[Entity]^-[System],[Entity]^-[SolvePattern],[Entity]^-[Node],[Entity]^-[Link],[Entity]^-[Flow])

## Children

 * [Flow](Flow.md)
 * [Link](Link.md)
 * [Node](Node.md)
 * [SolvePattern](SolvePattern.md)
 * [System](System.md)
 * [Unit](Unit.md)

## Referenced by Class

 *  **None** *[entity](entity.md)*  <sub>0..\*</sub>  **[Entity](Entity.md)**

## Attributes


### Own

 * [id](id.md)  <sub>1..1</sub>
     * Description: Unique identifier of the entity
     * Range: [Integer](types/Integer.md)
 * [name](name.md)  <sub>0..1</sub>
     * Description: Name of the entity
     * Range: [String](types/String.md)
 * [description](description.md)  <sub>0..1</sub>
     * Description: Description of the entity
     * Range: [String](types/String.md)
