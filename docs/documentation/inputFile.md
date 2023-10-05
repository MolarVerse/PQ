# The Input File {#inputFile}

[TOC]

## General

The concept of the input file is based on the definition of so-called "commands". A command in the input file can have one of the two following forms and is always case-insensitive:

~~~
    1) key = value;
~~~
~~~
    2) key = [value1, value2, ...];
~~~

<span style="color:red">**Note**</span>: The semicolon add the end of both command definitions is necessary, while the number of whitespace can be arbitrary at any position of the command as long as key and value are not split in parts.

Command definition 1) represents a single value command, whereas definition 2) can be considered as a list of input values to which will always be represented with `[]`.

#### Command Usage
Due to the use of `;` one line of the input file can contain multiple commands and blank lines will be ignored.

#### Comments
Every character following a `#` will be ignored. The `#` as a comment flag can be used also in all setup files - with some exceptions when contiguous input blocks are necessary.

#### Types of Input Values
In the following sections the types of the input values will be denoted via `{}`, where `{[]}` represents a list of types:

<div align="center">

|    Type    |          Description          |
| :--------: | :---------------------------: |
|   {int}    |            integer            |
|  {uint+}   |       positive integers       |
|   {uint}   | positive integers including 0 |
|  {double}  |    floating point numbers     |
|  {string}  |               -               |
|   {file}   |               -               |
|   {path}   |               -               |
| {pathFile} |     equal to {path/file}      |
|   {bool}   |          true/false           |

</div>

## Input Keys
<span style="color:red">**Note**</span>: Some of the following keys are necessary in the input file and are therefore marked with a `*`.

### General Keys

#### Jobtype

    jobtype* = {string} 

With the `jobtype` keyword the user can choose out of different engines to perform (not only) MD simulations.
    
<ins>Possible values are:</ins>

1) **mm-md** - represents a full molecular mechanics molecular dynamics simulation either performed via the `Guff` formalism or the `Amber force field`
   
2) **qm-md** - represents a full quantum mechanics molecular dynamics simulation. For more information see the [QM](#qmKeywords) keywords section.

3) **qm-rpmd** represents a full quantum mechanics ring polymer molecular dynamics simulation. For more information see the [Ring Polymer MD](#ringPolymerMD) keywords section

#### Timestep

    timestep* = {double} fs

With the `timestep` keyword the time step in fs of one molecular dynamics loop can be set.

#### Output_Freq

    output_freq = {uint}

The `output_freq` keyword sets the frequency (*i.e.* every n-th step) of how often the application should write into the output files. For a complete dry run without any output files it is also possible to set it to `0`.

### QM {#qmKeywords}

### Ring Polymer MD {#ringPolymerMD}

