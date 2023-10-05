# The Input File {#inputFile}

The concept of the input file is based on the definiton of so called "commands". A command in the input file can have one of the two following forms:

~~~
    1) key = value;
~~~
~~~
    2) key = [value1, value2, ...];
~~~

**Note**: The semicolon add the end of both command definitions is necessary, while the number of whitespace can be arbitrary at any position of the command as long as key and value are not split in parts.

Command definition 1) represents a single value command, whereas definition 2) can be considered as a list of input values to which will always be represented with `[]`.

##### Command Usage:
Due to the use of `;` one line of the input file can contain multiple commands and blank lines will be ignored.

##### Comments:
Every character following a `#` will be ignored. The `#` as a comment flag can be used also in all setup files - with some exceptions when contiguous input blocks are necessary.

#### Types of Input Values
In the following sections the types of the input values will be denoted via `{}`, where `{[]}` represents a list of types:

<div align="center">

|        Type        |          Description          |
| :----------------: | :---------------------------: |
|       {int}        |            integer            |
|       {uint}       |       positive integers       |
| {uint<sub>0</sub>} | positive integers including 0 |
|      {double}      |    floating point numbers     |
|      {string}      |               -               |
|       {file}       |               -               |
|       {path}       |               -               |
|     {pathFile}     |     equal to {path/file}      |
|       {bool}       |          true/false           |

</div>

