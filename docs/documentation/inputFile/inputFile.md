# The Input File {#inputFile}

[TOC]

## General

The concept of the input file is based on the definition of so-called "commands". A command in the input file can have one of the two following forms and is always case-insensitive:

    1) key = value;
    2) key = [value1, value2, ...];

<span style="color:red"><b>Note</b></span>: The semicolon add the end of both command definitions is necessary, while the number of whitespace can be arbitrary at any position of the command as long as key and value are not split in parts.

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
<span style="color:red"><b>Note</b></span>:: Some of the following keys are necessary in the input file and are therefore marked with a `*`. If there exists a default value for the possible values related to a key, they will be marked with `->` after the command.

### General Keys

#### Jobtype


    jobtype* = {string} 

>   With the `jobtype` keyword the user can choose out of different engines to perform (not only) MD simulations.
    
>  <ins>Possible values are:</ins>

>>      1) <b>mm-md</b> - represents a full molecular mechanics molecular dynamics simulation either performed via the `Guff` formalism or the `Amber force field`


>>      2) <b>qm-md</b> - represents a full quantum mechanics molecular dynamics simulation. For more information see the [QM](#qmKeywords) keywords section.


>>      3) <b>qm-rpmd</b> - represents a full quantum mechanics ring polymer molecular dynamics simulation. For more information see the [Ring Polymer MD](#ringPolymerMD) keywords section


#### Timestep

    timestep* = {double} fs

>   With the `timestep` keyword the time step in fs of one molecular dynamics loop can be set.

### General Output Keys

<span style="color:red"><b>Note</b></span>: The PIMD-QMCF application has a special naming convention for output files. For every job type a certain set of output files is written per default. If no output file names are given all prefixes of the output files will be named `default.<ext>`. If at least one of the output file keys was given in the input file - the program will determine the most common prefix (*i.e.* string before the first `.` character) and set it with the respective extension for all unspecified output files.

This feature enables easier post-processing of data and also easier deletion of all output files as PIMD-QMCF does not overwrite any existing output files and will throw an error.

#### Output_Freq

    output_freq = {uint} -> 1

>   The `output_freq` keyword sets the frequency (*i.e.* every n-th step) of how often the application should write into the output files. For a complete dry run without any output files it is also possible to set it to `0`.

>   *default value* = 1

#### File_Prefix

    file_prefix = {string}

>   The `file_prefix` keyword allows the user to set a common prefix name for all generated output files. The resulting names of the output files should be self-explanatory according to their unique file extension.

#### Output_File

    output_file = {file} -> "default.out"

>   The `output_file` keyword sets the name for the log file, in which all important information about the performed calculation can be found. 

>   *default value* = "default.out"

### MD Output File Keys

All of the following output files presented in the MD Output Files section are wriiten during calculations using MD related jobtypes.

#### Info_File

    info_file = {file} -> "default.info"

>   The `info_file` keyword sets the name for the info file, in which the most important physical properties of the last written step can be found.

>   *default value* = "default.info"

#### Energy_File

    energy_file = {file} -> "default.en"

>   The `energy_file` keyword sets the name for the energy file, in which the (almost) all important physical properties of the full simulation can be found.

>   *default value* = "default.en"

#### Rst_File

    rst_file = {file} -> "default.rst"

>   The `rst_file` keyword sets the name for the restart file, which contains all necessary information to restart (*i.e.* continue) the calculation from its timestamp.

>   *default value* = "default.rst"

#### Traj_File

    traj_file = {file} -> "default.xyz"

>   The `traj_file` keyword sets the name for the trajectory file of the atomic positions.

>   *default value* = "default.xyz"

#### Vel_File

    vel_file = {file} -> "default.vel"

>   The `vel_file` keyword sets the name for the trajectory file of the atomic velocities.

>   *default value* = "default.vel"

#### Force_File

    force_file = {file} -> "default.force"

>   The `force_file` keyword sets the name for the trajectory file of the atomic forces.

>   *default value* = "default.force"

#### Charge_File

    charge_file = {file} -> "default.chrg"

>   The `charge_file` keyword sets the name for the trajectory file of the atomic partial charges.

>   *default value* = "default.chrg"

### QM Keys {#qmKeywords}

### Ring Polymer MD Keys {#ringPolymerMD}

