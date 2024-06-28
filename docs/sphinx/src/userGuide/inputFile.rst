.. _inputFile:

##############
The Input File
##############

.. .. toctree::
   :maxdepth: 4
   :caption: Contents:

*******
General
*******

The concept of the input file is based on the definition of so-called "commands". A command in the input file can have one of the two following forms and is always case-insensitive:

    1) key = value;
    2) key = [value1, value2, ...];

.. Note::
    The semicolon add the end of both command definitions is necessary, while the number of whitespace can be arbitrary at any position of the command as long as key and value are not split in parts.

Command definition 1) represents a single value command, whereas definition 2) can be considered as a list of input values to which will always be represented with :code:`[]`.

Command Usage
=============

Due to the use of :code:`;` one line of the input file can contain multiple commands and blank lines will be ignored.

Comments
========

Every character following a :code:`#` will be ignored. The :code:`#` as a comment flag can be used also in all setup files - with some exceptions when contiguous input blocks are necessary.

Types of Input Values
=====================

In the following sections the types of the input values will be denoted via :code:`{}`, where :code:`{[]}` represents a list of types:

+-------------+-------------------------------+
|    Type     |          Description          |
+=============+===============================+
|   {int}     |            integer            |
+-------------+-------------------------------+
|  {uint+}    |       positive integers       |
+-------------+-------------------------------+
|   {uint}    | positive integers including 0 |
+-------------+-------------------------------+
|  {double}   |    floating point numbers     |
+-------------+-------------------------------+
|  {string}   |              \-               |
+-------------+-------------------------------+
|   {file}    |              \-               |
+-------------+-------------------------------+
|   {path}    |              \-               |
+-------------+-------------------------------+
| {pathFile}  |     equal to {path/file}      |
+-------------+-------------------------------+
|   {bool}    |          true/false           |
+-------------+-------------------------------+
| {selection} |          selection            |
+-------------+-------------------------------+

.. _selectionType:

.. Note::
    The :code:`{selection}` type is used to select a specific atom or group of atoms. If the PQ software package was build including :code:`python3.12` dependencies, the user can apply the selection grammar defined in the `PQAnalysis package <https://molarverse.github.io/PQAnalysis/code/PQAnalysis.topology.selection.html>`_. However, if PQ was compiled without these dependencies it is possible to index *via* the atomic indices starting from 0. If more than one atom index should be selected, the user can give a list of indices like :code:`{0, 1, 2}`. If a range of atom indices should be selected the user can use the following syntax :code:`{0-5, 10-15}` or :code:`{0..5, 10-15}` or :code:`{0..5, 10..15}`, where all would be equivalent to :code:`{0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15}`.

Input Keys
==========

.. Note::
    Some of the following keys are necessary in the input file and are therefore marked with a :code:`*`. If there exists a default value for the possible options related to a key, they will be marked with :code:`->` after the command.

************
General Keys
************

Jobtype
=======

.. admonition:: Key
    :class: tip

    jobtype* = {string} 

With the :code:`jobtype` keyword the user can choose out of different engines to perform (not only) MD simulations.

Possible options are:

   1) **mm-md** - represents a full molecular mechanics molecular dynamics simulation either performed via the :code:`Guff` formalism or the :code:`Amber force field`

   2) **qm-md** - represents a full quantum mechanics molecular dynamics simulation. For more information see the :ref:`qmKeys` keywords section.

   3) **qm-rpmd** - represents a full quantum mechanics ring polymer molecular dynamics simulation. For more information see the :ref:`ringPolymerMDKeys` keywords section

   4) **qmmm-md** - represents a hybrid quantum mechanics - molecular mechanics molecular dynamics simulation. (Not implemented yet)

   5) **opt** - represents a geometry optimization calculation. (Not implemented yet)


Timestep
========

.. admonition:: Key
    :class: tip

    timestep* = {double} fs

With the :code:`timestep` keyword the time step in :code:`fs` of one molecular dynamics loop can be set.

NStep
=====

.. admonition:: Key
    :class: tip

    nstep* = {uint+}

The :code:`ǹstep` keyword sets the total amount of MD steps to be performed within the next simulation run.

Integrator
==========

.. admonition:: Key
    :class: tip

    integrator = {string} -> "v-verlet"

With the :code:`integrator` keyword the user can choose the integrator type which should be applied.

Possible options are:

   1) **v-verlet** (default) - represents the Velocity-Verlet integrator 

Virial
======

.. admonition:: Key
    :class: tip

    virial = {string} -> "molecular"

With the :code:`virial` keyword the user can control if an intramolecular virial correction should be applied on the basis of molecular units definitions form the moldescriptor file.

Possible options are:

   1) **molecular** (default) - to the resulting virial from the force contributions an intramolecular correction will be applied.

   2) **atomic** - no intramolecular correction to the resulting virial will be applied

Start_File
==========

.. admonition:: Key
    :class: tip

    start_file* = {file}

The :code:`start_file` keyword sets the name of the start file for an MD simulation of any kind.

RPMD_Start_File
===============

.. admonition:: Key
    :class: tip

    rpmd_start_file = {file}

The :code:`rpmd_start_file` keyword is used to continue a ring polymer MD simulation containing positions, velocities and forces of all atoms of each bead of the ring polymer.

General Output Keys
===================

.. Note::
    The PQ application has a special naming convention for output files. For every job type a certain set of output files is written per default. If no output file names are given all prefixes of the output files will be named :code:`default.<ext>`. If at least one of the output file keys was given in the input file - the program will determine the most common prefix (*i.e.* string before the first :code:`.` character) and set it with the respective extension for all unspecified output files.

    This feature enables easier post-processing of data and also easier deletion of all output files as PQ does not overwrite any existing output files and will throw an error.

Output_Freq
===========

.. admonition:: Key
    :class: tip

    output_freq = {uint} -> 1

The :code:`output_freq` keyword sets the frequency (*i.e.* every n-th step) of how often the application should write into the output files. For a complete dry run without any output files it is also possible to set it to :code:`0`.

.. centered:: *default value* = 1

File_Prefix
===========

.. admonition:: Key
    :class: tip

    file_prefix = {string}

The :code:`file_prefix` keyword allows the user to set a common prefix name for all generated output files. The resulting names of the output files should be self-explanatory according to their unique file extension.

Output_File
===========

    output_file = {file} -> "default.out"

The :code:`output_file` keyword sets the name for the log file, in which all important information about the performed calculation can be found. 

.. centered:: *default value* = "default.out"

*******************
MD Output File Keys
*******************

All of the following output files presented in the MD Output Files section are wriiten during calculations using MD related jobtypes.

Info_File
=========

.. admonition:: Key
    :class: tip

    info_file = {file} -> "default.info"

The :code:`info_file` keyword sets the name for the info file, in which the most important physical properties of the last written step can be found.

.. centered:: *default value* = "default.info"

Energy_File
===========

.. admonition:: Key
    :class: tip

    energy_file = {file} -> "default.en"

The :code:`energy_file` keyword sets the name for the energy file, in which the (almost) all important physical properties of the full simulation can be found.

.. centered:: *default value* = "default.en"

Instant_Energy_File
===================

.. admonition:: Key
    :class: tip

    instant_energy_file = {file} -> "default.instant_en"

The :code:`instant_energy_file` keyword sets the name for the instant energy file, in which the energy of the system at each step can be found.

.. centered:: *default value* = "default.instant_en"

Rst_File
========

.. admonition:: Key
    :class: tip

    rst_file = {file} -> "default.rst"

The :code:`rst_file` keyword sets the name for the restart file, which contains all necessary information to restart (*i.e.* continue) the calculation from its timestamp.

.. centered:: *default value* = "default.rst"

Traj_File
=========

.. admonition:: Key
    :class: tip

    traj_file = {file} -> "default.xyz"

The :code:`traj_file` keyword sets the name for the trajectory file of the atomic positions.

.. centered:: *default value* = "default.xyz"

Vel_File
========

.. admonition:: Key
    :class: tip

    vel_file = {file} -> "default.vel"

The :code:`vel_file` keyword sets the name for the trajectory file of the atomic velocities.

.. centered:: *default value* = "default.vel"

Force_File
==========

.. admonition:: Key
    :class: tip

    force_file = {file} -> "default.force"

The :code:`force_file` keyword sets the name for the trajectory file of the atomic forces.

.. centered:: *default value* = "default.force"

Charge_File
===========

.. admonition:: Key
    :class: tip

    charge_file = {file} -> "default.chrg"

The :code:`charge_file` keyword sets the name for the trajectory file of the atomic partial charges.

.. centered:: *default value* = "default.chrg"

Momentum_File
=============

.. admonition:: Key
   :class: tip

    momentum_file = {file} -> "default.mom"

The :code:`momentum_file` keyword sets the name for output file containing the total linear momentum of the system, the individual box momenta in each direction as well as the corresponding angular momenta.

.. centered:: *default value* = "default.mom"

Virial_File
===========

.. admonition:: Key
    :class: tip

    virial_file = {file} -> "default.vir"

The :code:`virial_file` keyword sets the name for the output file containing the virial tensor of the system.

.. centered:: *default value* = "default.vir"

Stress_File
===========

.. admonition:: Key
    :class: tip

    stress_file = {file} -> "default.stress"

The :code:`stress_file` keyword sets the name for the output file containing the stress tensor of the system.

.. centered:: *default value* = "default.stress"

Box_File
========

.. admonition:: Key
    :class: tip

    box_file = {file} -> "default.box"

The :code:`box_file` keyword sets the name for the output file containing the lattice parameters a, b, c, :math:`\alpha`, :math:`\beta`, :math:`\gamma`.

.. centered:: *default value* = "default.box"

*********************
RPMD Output File Keys
*********************

All of the following output files presented in the RPMD Output Files section are wriiten during calculations using ring polymer MD related jobtypes. These files represents the trajectories of all individual beads.

RPMD_Restart_File
=================

.. admonition:: Key
    :class: tip

    rpmd_restart_file = {file} -> "default.rpmd.rst"

The :code:`rpmd_restart_file` keyword sets the name for the ring polymer restart file, which contains all necessary information to restart (*i.e.* continue) the calculation from its timestamp.

.. centered:: *default value* = "default.rpmd.rst"

RPMD_Traj_File
==============

.. admonition:: Key
    :class: tip

    rpmd_traj_file = {file} -> "default.rpmd.xyz"

The :code:`rpmd_traj_file` keyword sets the name for the file containing positions of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.xyz"

RPMD_Vel_File
=============

.. admonition:: Key
    :class: tip

    rpmd_vel_file = {file} -> "default.rpmd.vel"

The :code:`rpmd_vel_file` keyword sets the name for the file containing velocities of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.vel"

RPMD_Force_File
===============

.. admonition:: Key
    :class: tip

    rpmd_force_file = {file} -> "default.rpmd.force"

The :code:`rpmd_force_file` keyword sets the name for the file containing forces of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.force"

RPMD_Charge_File
================

.. admonition:: Key
    :class: tip

    rpmd_charge_file = {file} -> "default.rpmd.chrg"

The :code:`rpmd_charge_file` keyword sets the name for the file containing partial charges of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.chrg"

RPMD_Energy_File
================

.. admonition:: Key
    :class: tip

    rpmd_energy_file = {file} -> "default.rpmd.en"

The :code:`rpmd_energy_file` keyword sets the name for the file containing relevant energy data for each ring polymer bead of the simulation.

.. centered:: *default value* = "default.rpmd.en"

***********************
Input (Setup) File Keys
***********************

In order to setup certain calculations additional input files have to be used. The names of these files have to be specified in the input file. For further information about these input files can be found in the :ref:`setupFiles` section.

Moldesctiptor_File
==================

.. admonition:: Key
    :class: tip

    moldescriptor_file = {file} -> "moldescriptor.dat"

.. centered:: *default value* = "moldescriptor.dat"

Guff_File
=========

.. admonition:: Key
    :class: tip

    guff_file = {file} -> "guff.dat"

.. centered:: *default value* = "guff.dat"

Topology_File
=============

.. admonition:: Key
    :class: tip

    topology_file = {file}

Parameter_File
==============

.. admonition:: Key
    :class: tip

    parameter_file = {file}

Intra-NonBonded_File
====================

.. admonition:: Key
    :class: tip

    intra-nonbonded_file = {file}

*******************
Simulation Box Keys
*******************

Density
=======

.. admonition:: Key
    :class: tip

    density = {double} kgL⁻¹

With the :code:`density` keyword the box dimension of the system can be inferred from the total mass of the simulation box.

.. Note::
    This keyword implies that the simulation box has cubic shape. Furthermore, the :code:`density` keyword will be ignored if in the start file of a simulation any box information is given.

RCoulomb
========

.. admonition:: Key
    :class: tip


    rcoulomb = {double} :math:`\mathrm{\mathring{A}}` -> 12.5 :math:`\mathrm{\mathring{A}}`

With the :code:`rcoulomb` keyword the radial cut-off in :math:`\mathrm{\mathring{A}}` of Coulomb interactions for MM-MD type simulations can be set. If pure QM-MD type simulations are applied this keyword will be ignored and the value will be set to 0 :math:`\mathrm{\mathring{A}}`.

.. centered:: *default value* = 12.5 :math:`\mathrm{\mathring{A}}` (for MM-MD type simulations)

Init_Velocities
===============

.. admonition:: Key
    :class: tip

    init_velocities = {bool} -> false

To initialize the velocities of the system according to the target temperature with a Boltzmann distribution the user has to set the :code:`init_velocities` to true.

Possible options are:

   1) **false** (default) - velocities are taken from start file

   2) **true** - velocities are initialized according to a Boltzmann distribution at the target temperature.

*************************
Temperature Coupling Keys
*************************

Temperature
===========

.. admonition:: Key
    :class: tip

    temp = {double} K

With the :code:`temp` keyword the target temperature in :code:`K` of the system can be set. 

.. Note::
    This keyword is not restricted to the use of any temperature coupling method, as it is used *e.g.* also for the initialization of Boltzmann distributed velocities or the reset of the system temperature.

Start_Temperature
=================

.. admonition:: Key
    :class: tip

    start_temp = {double} K

With the :code:`start_temp` keyword the initial temperature in :code:`K` of the system can be set. If a value is given the PQ application will perform a temperature ramping from the :code:`start_temp` to the :code:`temp` value.

End_Temperature
===============

.. admonition:: Key
    :class: tip

    end_temp = {double} K

The :code:`end_temp` keyword is a synonym for the :code:`temp` keyword and can be used to set the target temperature of the system. It cannot be used in combination with the :code:`temp` keyword.

Temperature_Ramp_Steps
======================

.. admonition:: Key
    :class: tip

    temp_ramp_steps = {uint+}

With the :code:`temp_ramp_steps` keyword the user can specify the number of steps for the temperature ramping from the :code:`start_temp` to the :code:`temp` value. If no starting temperature is given the keyword will be ignored. If a starting temperature is given and this keyword is omitted the temperature ramping will be performed over the full simulation time.

.. centered:: *default value* = full simulation time

Temperature_Ramp_Frequency
==========================

.. admonition:: Key
    :class: tip

    temp_ramp_freq = {uint+} -> 1

With the :code:`temp_ramp_freq` keyword the user can specify the frequency of the temperature ramping from the :code:`start_temp` to the :code:`temp` value. If no starting temperature is given the keyword will be ignored. If a starting temperature is given and this keyword is omitted the temperature ramping will be performed, so that each step the temperature is increased by the same value.

.. centered:: *default value* = 1 step

Thermostat
==========
.. TODO: reference manual

.. admonition:: Key
    :class: tip

    thermostat = {string} -> "none"

With the :code:`thermostat` keyword the temperature coupling method can be chosen.

Possible options are:

   1) **none** (default) - no thermostat is set, hence {N/µ}{p/V}E settings are applied.

   2) **berendsen** - the Berendsen weak coupling thermostat

   3) **velocity_rescaling** - the stochastic velocity rescaling thermostat

   4) **langevin** - temperature coupling *via* stochastic Langevin dynamics

   5) **nh-chain** - temperature coupling *via* Nose Hoover extended Lagrangian 

T_Relaxation
============

This keyword is used in combination with the Berendsen and velocity rescaling thermostat.

.. admonition:: Key
    :class: tip

    t_relaxation = {double} ps -> 0.1 ps

With the :code:`t_relaxation` keyword the relaxation time in :code:`ps` (*i.e.* :math:`\tau`) of the Berendsen or stochastic velocity rescaling thermostat is set.

.. centered:: *default value* = 0.1 ps

Friction
========

.. admonition:: Key
    :class: tip

    friction = {double} ps⁻¹ -> 0.1 ps⁻¹

With the :code:`friction` keyword the friction in :code:`ps⁻¹` applied in combination with the Langevin thermostat can be set.

.. centered:: *default value* = 0.1 ps⁻¹

NH-Chain_Length
===============

.. admonition:: Key
    :class: tip

    nh-chain_length = {uint+} -> 3

With the :code:`nh-chain_length` keyword the length of the chain for temperature control *via* an extended Nose-Hoover Lagrangian can be set.

.. centered:: *default value* = 3

Coupling_Frequency
==================

.. admonition:: Key
    :class: tip

    coupling_frequency = {double} cm⁻¹ -> 1000 cm⁻¹

With the :code:`coupling_frequency` keyword the coupling frequency of the Nose-Hoover chain in :code:`cm⁻¹` can be set.

.. centered:: *default value* = 1000 cm⁻¹

.. _pressureCouplingKeys:

**********************
Pressure Coupling Keys
**********************

Pressure
========

.. admonition:: Key
    :class: tip

    pressure = {double} bar

With the :code:`pressure` keyword the target pressure in :code:`bar` of the system can be set. 

.. Note::
    This keyword is only used if a manostat for controlling the pressure is explicitly defined.

Manostat
========
.. TODO: reference manual

.. admonition:: Key
    :class: tip

    manostat = {string} -> "none"

With the :code:`manostat` keyword the type of the pressure coupling can be chosen.

Possible options are:

   1) **none** (default) - no pressure coupling is applied (*i.e.* constant volume)

   2) **berendsen** - Berendsen weak coupling manostat

   3) **stochastic_rescaling** - stochastic cell rescaling manostat

P_Relaxation
============

This keyword is used in combination with the Berendsen and stochastic cell rescaling manostat.

.. admonition:: Key
    :class: tip

    p_relaxation = {double} ps -> 0.1 ps

With the :code:`p_relaxation` keyword the relaxation time in :code:`ps` (*i.e.* :math:`\tau`) of the Berendsen or stochastic cell rescaling manostat is set.

.. centered:: *default value* = 0.1 ps

Compressibility
===============

This keyword is used in combination with the Berendsen and stochastic cell rescaling manostat.

.. admonition:: Key
    :class: tip

    compressibility = {double} bar⁻¹ -> 4.591e-5 bar⁻¹

With the :code:`compressibility` keyword the user can specify the compressibility of the target system in :code:`bar⁻¹` for the Berendsen and stochastic cell rescaling manostat.

.. centered:: *default value* = 4.591e-5 bar⁻¹ (compressibility of water)

Isotropy
========

.. admonition:: Key
    :class: tip

    isotropy = {string} -> "isotropic"

With the :code:`isotropy` keyword the isotropy of the pressure coupling for all manostat types is controlled.

Possible options are:

   1) **isotropic** (default) - all axes are scaled with the same scaling factor

   2) **xy** - semi-isotropic settings, with axes :code:`x` and :code:`y` coupled isotropic

   3) **xz** - semi-isotropic settings, with axes :code:`x` and :code:`z` coupled isotropic

   4) **yz** - semi-isotropic settings, with axes :code:`y` and :code:`z` coupled isotropic

   5) **anisotropic** - all axes are coupled in an anisotropic way

   6) **full_anisotropic** - all axes are coupled in an anisotropic way and the box angles are also scaled

*******************
Reset Kinetics Keys
*******************

NScale
======

.. admonition:: Key
    :class: tip

    nscale = {uint} -> 0

With the :code:`nscale` keyword the user can specify the first :code:`n` steps in which the temperature is reset *via* a hard scaling approach to the target temperature.

.. Note::
    Resetting the temperature to the target temperature does imply also a subsequent reset of the total box momentum. Furthermore, resetting to the target temperature does not necessarily require a constant temperature ensemble setting.

.. centered:: *default value* = 0 (*i.e.* never)

FScale
======

.. admonition:: Key
    :class: tip

    fscale = {uint} -> nstep + 1

With the :code:`fscale` keyword the user can specify the frequency :code:`f` at which the temperature is reset *via* a hard scaling approach to the target temperature.

.. Note:: 
    Resetting the temperature to the target temperature does imply also a subsequent reset of the total box momentum. Furthermore, resetting to the target temperature does not necessarily require a constant temperature ensemble setting.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1 

NReset
======

.. admonition:: Key
    :class: tip

    nreset = {uint} -> 0

With the :code:`nreset` keyword the user can specify the first :code:`n` steps in which the total box momentum is reset.

.. centered:: *default value* = 0 (*i.e.* never)

FReset
======

.. admonition:: Key
    :class: tip

    freset = {uint} -> nstep + 1

With the :code:`freset` keyword the user can specify the frequency :code:`f` at which the total box momentum is reset.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1

NReset_Angular
==============

.. admonition:: Key
    :class: tip

    nreset_angular = {uint} -> 0

With the :code:`nreset_angular` keyword the user can specify the first :code:`n` steps in which the total angular box momentum is reset.

.. Danger::
    This setting should be used very carefully, since in periodic system a reset of the angular momentum can result in some very unphysical behavior.

.. centered:: *default value* = 0 (*i.e.* never)

FReset_Angular
==============

.. admonition:: Key
    :class: tip

    freset_angular = {uint} -> nstep + 1

With the :code:`freset_angular` keyword the user can specify the frequency :code:`f` at which the total angular box momentum is reset.

.. Danger::
    This setting should be used very carefully, since in periodic system a reset of the angular momentum can result in some very unphysical behavior.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1 

****************
Constraints Keys
****************

Shake
=====

.. admonition:: Key
    :class: tip

    shake = {string} -> "off"

With the :code:`shake` keyword it is possible to activate the SHAKE/RATTLE algorithm for bond constraints.

Possible options are:

   1) **off** (default) - no shake will be applied

   2) **on** - SHAKE for bond constraints defined in the :ref:`topologyFile` will be applied.

   3) **shake** - SHAKE for bond constraints defined in the :ref:`topologyFile` will be applied.

   4) **mshake** - M-SHAKE for bond constraints defined in a special :ref:`mshakeFile` will be applied. As the M-SHAKE algorithm is designed for the treatment of rigid body molecular units the general shake algorithm will be activated automatically along with the M-SHAKE algorithm. The shake bonds can be defined as usual in the :ref:`topologyFile` and if no SHAKE bonds are defined only the M-SHAKE algorithm will be applied (without any overhead)

Shake-Tolerance
===============

.. admonition:: Key
    :class: tip

    shake-tolerance = {double} -> 1e-8

With the :code:`shake-tolerance` keyword the user can specify the tolerance, with which the bond-length of the shaked bonds should converge.

.. centered:: *default value* = 1e-8

Shake-Iter
==========

.. admonition:: Key
    :class: tip

    shake-iter = {uint+} -> 20

With the :code:`shake-iter` keyword the user can specify the maximum number of iteration until the convergence of the bond-lengths should be reached within the shake algorithm.

.. centered:: *default value* = 20

Rattle-Tolerance
================

.. admonition:: Key
    :class: tip


    rattle-tolerance = {double} s⁻¹kg⁻¹ -> 1e4 s⁻¹kg⁻¹ 


With the :code:`rattle-tolerance` keyword the user can specify the tolerance in :code:`s⁻¹kg⁻¹`, with which the velocities of the shaked bonds should converge.

.. centered:: *default value* = 20 s⁻¹kg⁻¹

Rattle-Iter
===========

.. admonition:: Key
    :class: tip

    rattle-iter = {uint+} -> 20

With the :code:`rattle-iter` keyword the user can specify the maximum number of iteration until the convergence of the velocities of the shaked bond-lengths should be reached within the rattle algorithm.

.. centered:: *default value* = 20

Distance-Constraints
====================

.. admonition:: Key
    :class: tip

    distance-constraints = {string} -> "off"

With the :code:`distance-constraints` keyword it is possible to activate the distance constraints for the simulation. The distance constraints are defined in the :ref:`topologyFile`.

*******
MM Keys
*******

NonCoulomb
==========

.. admonition:: Key
    :class: tip

    noncoulomb = {string} -> "guff"

With the :code:`noncoulomb` keyword the user can specify which kind of [GUFF formalism](#guffdatFile) should be used for parsing the guff.dat input file. <span style="color:red"><b>Note</b></span>: This keyword is only considered if an MM-MD type simulation is requested and the force field is not turned on.

Possible options are:

   1) **guff** (default) - full GUFF formalism

   2) **lj** - Lennard Jones quick routine

   3) **buck** - Buckingham quick routine

   4) **morse** - Morse quick routine

ForceField
==========

.. admonition:: Key
    :class: tip

    forcefield = {string} -> "off"

With the :code:`forcefield` keyword the user can switch from the GUFF formalism to force field type simulation (For details see Reference Manual).

Possible options are:

   1) **off** (default) - GUFF formalism is applied

   2) **on** - full force field definition is applied

   3) **bonded** - non bonded interaction are described *via* GUFF formalism and bonded interactions *via* force field approach

*********************
Long Range Correction
*********************

Long_Range
==========

.. admonition:: Key
    :class: tip

    long_range = {string} -> "none"

With the :code:`long_range` correction keyword the user can specify the type of <b>Coulombic<B> long range correction, which should be applied during the Simulation.

Possible options are:

   1) **none** (default) - no long range correction

   2) **wolf** - Wolf summation

Wolf_Param
==========
.. TODO: add unit and description

.. admonition:: Key
    :class: tip

    wolf_param = {double} -> 0.25 

.. centered:: *default value* = 0.25

.. _qmKeys:

*******
QM Keys
*******

QM_PROG
=======

.. admonition:: Key
    :class: tip

    qm_prog = {string}

With the :code:`qm_prog` keyword the external QM engine for any kind of QM MD simulation is chosen.

.. Note::
    This keyword is required for any kind of QM MD simulation!

Possible options are:

   1) **dftbplus**

   2) **pyscf**

   3) **turbomole**

QM_SCRIPT
=========

.. admonition:: Key
    :class: tip

    qm_script = {file}

With the :code:`qm_script` keyword the external executable to run the QM engine and to parse its output is chosen. All possible scripts can be found under `<https://github.com/MolarVerse/PQ/tree/main/src/QM/scripts>`_. Already the naming of the executables should hopefully be self-explanatory in order to choose the correct input executable name.

QM_SCRIPT_FULL_PATH
===================

.. admonition:: Key
    :class: tip

    qm_script_full_path = {pathFile}

.. attention::
   This keyword can not be used in conjunction with the :code:`qm_script` keyword! Furthermore, this keyword needs to be used in combination with any singularity or static build of PQ. For further details regarding the compilation/installation please refer to the :ref:`userG_installation` section.



With the :code:`qm_script_full_path` keyword the user can specify the full path to the external executable to run the QM engine and to parse its output. All possible scripts can be found under `<https://github.com/MolarVerse/PQ/tree/main/src/QM/scripts>`_. Already the naming of the executables should hopefully be self-explanatory in order to choose the correct input executable name.

QM_LOOP_TIME_LIMIT
==================

.. admonition:: Key
    :class: tip

    qm_loop_time_limit = {double} s -> -1 s

With the :code:`qm_loop_time_limit` keyword the user can specify the loop time limit in :code:`s` of all QM type calculations. If the time limit is reached the calculation will be stopped. Default value is -1 s, which means no time limit is set, and the calculation will continue until it is finished. In general all negative values will be interpreted as no time limit.

.. _ringPolymerMDKeys:

********************
Ring Polymer MD Keys
********************

RPMD_n_replica
==============

.. admonition:: Key
    :class: tip

    rpmd_n_replica = {uint+}

With the :code:`rpmd_n_replica` keyword the number of beads for a ring polymer MD simulation is controlled.

.. Note::
    This keyword is required for any kind of ring polymer MD simulation!

**********
QM/MM Keys
**********

QM_Center
=========

.. admonition:: Key
    :class: tip

    qm_center = {selection} -> 0

With the :code:`qm_center` keyword the user can specify the center of the QM region. The default selection is the first atom of the system (*i.e.* 0). For more information about the selection grammar see the `selectionType`_ section. The :code:`qm_center` if more than one atom is selected will be by default the center of mass of the selected atoms.

QM_Only_List
============

.. admonition:: Key
    :class: tip

    qm_only_list = {selection}

With the :code:`qm_only_list` keyword the user can specify a list of atoms which should be treated as QM atoms only. This means that these atoms can not leave the QM region during the simulation. For more information see the reference manual. For more information about the selection grammar see the `selectionType`_ section. By default no atom is selected.

MM_Only_List
============

.. admonition:: Key
    :class: tip

    mm_only_list = {selection}

With the :code:`mm_only_list` keyword the user can specify a list of atoms which should be treated as MM atoms only. This means that these atoms can not enter the QM region during the simulation. For more information see the reference manual. For more information about the selection grammar see the `selectionType`_ section. By default no atom is selected.

QM_Charges
==========

.. admonition:: Key
    :class: tip

    qm_charges = {string} -> "off"

With the :code:`qm_charges` keyword the user can specify the charge model for the QM atoms. If the :code:`qm_charges` keyword is set to :code:`off` the charges of the QM atoms are taken from the MM model applied. If the :code:`qm_charges` keyword is set to :code:`on` the charges of the QM atoms are taken from the QM calculation.

QM_Core_Radius
==============

.. admonition:: Key
    :class: tip

    qm_core_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}}`

With the :code:`qm_core_radius` keyword the user can specify the core radius in :math:`\mathrm{\mathring{A}}` around the :code:`qm_center`. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that the core radius is not set and only explicit QM atoms are used for the QM region.

QMMM_Layer_Radius
=================

.. admonition:: Key
    :class: tip

    qmmm_layer_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}`

With the :code:`qmmm_layer_radius` keyword the user can specify the layer radius in :math:`\mathrm{\mathring{A}}` around the :code:`qm_center`. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that no special QM/MM treatment is applied.

QMMM_Smoothing_Radius
=====================

.. admonition:: Key
    :class: tip

    qmmm_smoothing_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}`

With the :code:`qmmm_smoothing_radius` keyword the user can specify the smoothing radius in :math:`\mathrm{\mathring{A}}` of the QM atoms. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that the smoothing radius is not set and no smoothing is applied.

**************
Cell List Keys
**************

Cell-List
=========

.. admonition:: Key
    :class: tip

    cell-list = {string} -> "off"

With the :code:`cell-list` the user can activate a cell-list approach to calculate the pair-interactions in MM-MD simulations (no effect in pure QM-MD type simulations).

Possible options are:

   1) **off** (default) - brute force routine

   2) **on** - cell list approach is applied

Cell-Number
===========

.. admonition:: Key
    :class: tip

    cell-number = {uint+} -> 7

With the :code:`cell-number` keyword the user can set the number of cells in each direction in which the simulation box will be split up (*e.g.* cell-number = 7 -> total cells = 7x7x7)

.. centered:: *default value* = 7
