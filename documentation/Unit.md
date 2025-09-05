
# Class: Unit



URI: [file:///ines-core.yaml/Unit](file:///ines-core.yaml/Unit)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20unit%200..*>[Unit&#124;efficiency:float%20%3F;units_existing:float%20%3F;conversion_method:conversion_method_enum%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[NodeToUnit]-%20sink%200..1>[Unit],[UnitToNode]-%20source%200..1>[Unit],[Unit]uses%20-.->[HasInvestments],[Entity]^-[Unit],[UnitToNode],[NodeToUnit],[HasInvestments],[Entity],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20unit%200..*>[Unit&#124;efficiency:float%20%3F;units_existing:float%20%3F;conversion_method:conversion_method_enum%20%3F;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[NodeToUnit]-%20sink%200..1>[Unit],[UnitToNode]-%20source%200..1>[Unit],[Unit]uses%20-.->[HasInvestments],[Entity]^-[Unit],[UnitToNode],[NodeToUnit],[HasInvestments],[Entity],[Database])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Uses Mixin

 *  mixin: [HasInvestments](HasInvestments.md)

## Referenced by Class

 *  **None** *[➞unit](database__unit.md)*  <sub>0..\*</sub>  **[Unit](Unit.md)**
 *  **None** *[➞sink](nodeToUnit__sink.md)*  <sub>0..1</sub>  **[Unit](Unit.md)**
 *  **None** *[➞source](unitToNode__source.md)*  <sub>0..1</sub>  **[Unit](Unit.md)**

## Attributes


### Own

 * [efficiency](efficiency.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [units_existing](units_existing.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [➞conversion_method](unit__conversion_method.md)  <sub>0..1</sub>
     * Description: constant_efficiency
     * Range: [conversion_method_enum](conversion_method_enum.md)

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
