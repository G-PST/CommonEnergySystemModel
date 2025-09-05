
# Class: Database



URI: [file:///ines-core.yaml/Database](file:///ines-core.yaml/Database)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[UnitToNode],[Unit],[System],[Storage],[SolvePattern],[NodeToUnit],[Link],[System]<system%200..*-++[Database],[SolvePattern]<solve_pattern%200..*-++[Database],[Link]<link%200..*-++[Database],[UnitToNode]<unit__to_node%200..*-++[Database],[NodeToUnit]<node__to_unit%200..*-++[Database],[Unit]<unit%200..*-++[Database],[Commodity]<commodity%200..*-++[Database],[Storage]<storages%200..*-++[Database],[Balance]<balances%200..*-++[Database],[Commodity],[Balance])](https://yuml.me/diagram/nofunky;dir:TB/class/[UnitToNode],[Unit],[System],[Storage],[SolvePattern],[NodeToUnit],[Link],[System]<system%200..*-++[Database],[SolvePattern]<solve_pattern%200..*-++[Database],[Link]<link%200..*-++[Database],[UnitToNode]<unit__to_node%200..*-++[Database],[NodeToUnit]<node__to_unit%200..*-++[Database],[Unit]<unit%200..*-++[Database],[Commodity]<commodity%200..*-++[Database],[Storage]<storages%200..*-++[Database],[Balance]<balances%200..*-++[Database],[Commodity],[Balance])

## Attributes


### Own

 * [➞balances](database__balances.md)  <sub>0..\*</sub>
     * Range: [Balance](Balance.md)
 * [➞storages](database__storages.md)  <sub>0..\*</sub>
     * Range: [Storage](Storage.md)
 * [➞commodity](database__commodity.md)  <sub>0..\*</sub>
     * Range: [Commodity](Commodity.md)
 * [➞unit](database__unit.md)  <sub>0..\*</sub>
     * Range: [Unit](Unit.md)
 * [➞node__to_unit](database__node__to_unit.md)  <sub>0..\*</sub>
     * Range: [NodeToUnit](NodeToUnit.md)
 * [➞unit__to_node](database__unit__to_node.md)  <sub>0..\*</sub>
     * Range: [UnitToNode](UnitToNode.md)
 * [➞link](database__link.md)  <sub>0..\*</sub>
     * Range: [Link](Link.md)
 * [➞solve_pattern](database__solve_pattern.md)  <sub>0..\*</sub>
     * Range: [SolvePattern](SolvePattern.md)
 * [➞system](database__system.md)  <sub>0..\*</sub>
     * Range: [System](System.md)
