
# Class: Link



URI: [file:///ines-core.yaml/Link](file:///ines-core.yaml/Link)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[Node]<node_B%200..1-%20[Link&#124;efficiency:float%20%3F;capacity:float%20%3F;links_existing:float%20%3F;investment_cost:float%20%3F;transfer_method:transfer_method_enum%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]<node_A%200..1-%20[Link],[Database]++-%20link%200..*>[Link],[Link]uses%20-.->[HasInvestments],[Entity]^-[Link],[HasInvestments],[Entity],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Node],[Node]<node_B%200..1-%20[Link&#124;efficiency:float%20%3F;capacity:float%20%3F;links_existing:float%20%3F;investment_cost:float%20%3F;transfer_method:transfer_method_enum%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Node]<node_A%200..1-%20[Link],[Database]++-%20link%200..*>[Link],[Link]uses%20-.->[HasInvestments],[Entity]^-[Link],[HasInvestments],[Entity],[Database])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Uses Mixin

 *  mixin: [HasInvestments](HasInvestments.md)

## Referenced by Class

 *  **None** *[➞link](database__link.md)*  <sub>0..\*</sub>  **[Link](Link.md)**

## Attributes


### Own

 * [node_A](node_A.md)  <sub>0..1</sub>
     * Description: First node of a bidirectional link
     * Range: [Node](Node.md)
 * [node_B](node_B.md)  <sub>0..1</sub>
     * Description: Second node of a bidirectional link
     * Range: [Node](Node.md)
 * [Link➞efficiency](Link_efficiency.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [capacity](capacity.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [links_existing](links_existing.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [investment_cost](investment_cost.md)  <sub>0..1</sub>
     * Description: Price (in currency_year denomination) per kW
     * Range: [Float](types/Float.md)
 * [➞transfer_method](link__transfer_method.md)  <sub>0..1</sub>
     * Description: regular_linear
     * Range: [transfer_method_enum](transfer_method_enum.md)

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
