
# Class: HasFlow



URI: [file:///ines-core.yaml/HasFlow](file:///ines-core.yaml/HasFlow)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Storage]uses%20-.->[HasFlow&#124;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F],[Balance]uses%20-.->[HasFlow],[Storage],[Balance])](https://yuml.me/diagram/nofunky;dir:TB/class/[Storage]uses%20-.->[HasFlow&#124;flow_annual:float%20%3F;flow_profile:float%20*;flow_scaling_method:flow_scaling_method_enum%20%3F],[Balance]uses%20-.->[HasFlow],[Storage],[Balance])

## Mixin for

 * [Balance](Balance.md) (mixin) 
 * [Storage](Storage.md) (mixin) 

## Referenced by Class


## Attributes


### Own

 * [➞flow_annual](hasFlow__flow_annual.md)  <sub>0..1</sub>
     * Range: [Float](types/Float.md)
 * [➞flow_profile](hasFlow__flow_profile.md)  <sub>0..\*</sub>
     * Range: [Float](types/Float.md)
 * [➞flow_scaling_method](hasFlow__flow_scaling_method.md)  <sub>0..1</sub>
     * Description: use_profile_directly, scale_to_annual
     * Range: [flow_scaling_method_enum](flow_scaling_method_enum.md)
