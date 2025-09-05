
# Class: Solve_pattern



URI: [file:///ines-core.yaml/SolvePattern](file:///ines-core.yaml/SolvePattern)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20solve_pattern%200..*>[SolvePattern&#124;solve_mode:solve_model_enum%20%3F;start_time:integer%20%3F;duration:integer%20%3F;time_resolution:string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Entity]^-[SolvePattern],[Entity],[Database])](https://yuml.me/diagram/nofunky;dir:TB/class/[Database]++-%20solve_pattern%200..*>[SolvePattern&#124;solve_mode:solve_model_enum%20%3F;start_time:integer%20%3F;duration:integer%20%3F;time_resolution:string%20%3F;id(i):integer;name(i):string%20%3F;description(i):string%20%3F],[Entity]^-[SolvePattern],[Entity],[Database])

## Parents

 *  is_a: [Entity](Entity.md) - abstract high-level class

## Referenced by Class

 *  **None** *[➞solve_pattern](database__solve_pattern.md)*  <sub>0..\*</sub>  **[SolvePattern](SolvePattern.md)**

## Attributes


### Own

 * [➞solve_mode](solvePattern__solve_mode.md)  <sub>0..1</sub>
     * Description: single_solve
     * Range: [solve_model_enum](solve_model_enum.md)
 * [➞start_time](solvePattern__start_time.md)  <sub>0..1</sub>
     * Range: [Integer](types/Integer.md)
 * [➞duration](solvePattern__duration.md)  <sub>0..1</sub>
     * Range: [Integer](types/Integer.md)
 * [➞time_resolution](solvePattern__time_resolution.md)  <sub>0..1</sub>
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
