
# Class: HasInvestments



URI: [file:///ines-core.yaml/HasInvestments](file:///ines-core.yaml/HasInvestments)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit]uses%20-.->[HasInvestments&#124;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F],[Storage]uses%20-.->[HasInvestments],[Link]uses%20-.->[HasInvestments],[Unit],[Storage],[Link])](https://yuml.me/diagram/nofunky;dir:TB/class/[Unit]uses%20-.->[HasInvestments&#124;investment_method:investment_method_enum%20%3F;interest_rate:float%20%3F;lifetime:float%20%3F],[Storage]uses%20-.->[HasInvestments],[Link]uses%20-.->[HasInvestments],[Unit],[Storage],[Link])

## Mixin for

 * [Link](Link.md) (mixin) 
 * [Storage](Storage.md) (mixin) 
 * [Unit](Unit.md) (mixin) 

## Referenced by Class


## Attributes


### Own

 * [➞investment_method](hasInvestments__investment_method.md)  <sub>0..1</sub>
     * Description: not_allowed, no_limits
     * Range: [investment_method_enum](investment_method_enum.md)
 * [➞interest_rate](hasInvestments__interest_rate.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [➞lifetime](hasInvestments__lifetime.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
