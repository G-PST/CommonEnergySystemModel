
# Class: Flow



URI: [file:///ines-core.yaml/Flow](file:///ines-core.yaml/Flow)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[UnitToNode],[NodeToUnit],[HasProfiles],[Flow&#124;source:string%20%3F;sink:string%20%3F;capacity:float%20%3F;other_operational_cost:float%20%3F;investment_cost:float%20%3F;profile_method:profile_method_enum%20%3F;profile_limit_upper:float%20*;profile_limit_lower:float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F]uses%20-.->[HasProfiles],[Flow]^-[UnitToNode],[Flow]^-[NodeToUnit],[Entity]^-[Flow],[Entity])](https://yuml.me/diagram/nofunky;dir:TB/class/[UnitToNode],[NodeToUnit],[HasProfiles],[Flow&#124;source:string%20%3F;sink:string%20%3F;capacity:float%20%3F;other_operational_cost:float%20%3F;investment_cost:float%20%3F;profile_method:profile_method_enum%20%3F;profile_limit_upper:float%20*;profile_limit_lower:float%20*;id(i):integer;name(i):string%20%3F;description(i):string%20%3F]uses%20-.->[HasProfiles],[Flow]^-[UnitToNode],[Flow]^-[NodeToUnit],[Entity]^-[Flow],[Entity])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Uses Mixin

 *  mixin: [HasProfiles](HasProfiles.md)

## Children

 * [NodeToUnit](NodeToUnit.md)
 * [UnitToNode](UnitToNode.md)

## Referenced by Class


## Attributes


### Own

 * [source](source.md)  <sub>0..1</sub>
     * Description: Source of the unidirectional flow
     * Range: [String](types/String.md)
 * [sink](sink.md)  <sub>0..1</sub>
     * Description: Sink of the unidirectional flow
     * Range: [String](types/String.md)
 * [capacity](capacity.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [other_operational_cost](other_operational_cost.md)  <sub>0..1</sub>
     * Description: Cost (in currency_year denomination) per unit of the product flowing through
     * Range: [Float](types/Float.md)
 * [investment_cost](investment_cost.md)  <sub>0..1</sub>
     * Description: Price (in currency_year denomination) per kW
     * Range: [Float](types/Float.md)

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

### Mixed in from HasProfiles:

 * [➞profile_method](hasProfiles__profile_method.md)  <sub>0..1</sub>
     * Description: upper_limit, lower_limit
     * Range: [profile_method_enum](profile_method_enum.md)

### Mixed in from HasProfiles:

 * [➞profile_limit_upper](hasProfiles__profile_limit_upper.md)  <sub>0..\*</sub>
     * Range: [Float](types/Float.md)

### Mixed in from HasProfiles:

 * [➞profile_limit_lower](hasProfiles__profile_limit_lower.md)  <sub>0..\*</sub>
     * Range: [Float](types/Float.md)
