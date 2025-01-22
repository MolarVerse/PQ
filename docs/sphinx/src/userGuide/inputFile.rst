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

    1. key = value;
    2. key = [value1, value2, ...];

.. Note::
    The semicolon add the end of both command definitions is necessary, while the number of whitespace can be arbitrary at any position of the command as long as key and value are not split in parts.

Command definition 1. represents a single value command, whereas definition 2. can be considered as a list of input values to which will always be represented with ``[]``.

Command Usage
=============

Due to the use of ``;`` one line of the input file can contain multiple commands and blank lines will be ignored.

Comments
========

Every character following a ``#`` will be ignored. The ``#`` as a comment flag can be used also in all setup files - with some exceptions when contiguous input blocks are necessary.

Types of Input Values
=====================

In the following sections the types of the input values will be denoted via ``{}``, where ``{[]}`` represents a list of types:

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
    The ``{selection}`` type is used to select a specific atom or group of atoms. If the PQ software package was build including ``python3.12`` dependencies, the user can apply the selection grammar defined in the `PQAnalysis package <https://molarverse.github.io/PQAnalysis/code/PQAnalysis.topology.selection.html>`_. However, if PQ was compiled without these dependencies it is possible to index *via* the atomic indices starting from 0. If more than one atom index should be selected, the user can give a list of indices like ``{0, 1, 2}``. If a range of atom indices should be selected the user can use the following syntax ``{0-5, 10-15}`` or ``{0..5, 10-15}`` or ``{0..5, 10..15}``, where all would be equivalent to ``{0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15}``.

Input Keys
==========

.. Note::
    Some of the following keys are necessary in the input file and are therefore marked with a ``*``. If there exists a default value for the possible options related to a key, they will be marked with ``->`` after the command. Units of the values are given after the ``{}``.

************
General Keys
************

.. _jobtype:

Jobtype
=======

.. admonition:: Key
    :class: tip

    jobtype* = {string} 

With the ``jobtype`` keyword the user can choose from different engines to perform (not only) MD simulations.

Possible options are:

   1. **mm-md** - Represents a full molecular mechanics molecular dynamics simulation performed via the :ref:`GUFF <guffdatFile>` formalism and/or the ``Amber force field``. Respective input file keys are found in the :ref:`mmKeys` section.

   2. **qm-md** - Represents a full quantum mechanics molecular dynamics simulation. For more information see the :ref:`qmKeys` section.

   3. **qm-rpmd** - Represents a full quantum mechanics ring polymer molecular dynamics simulation. For more information see the :ref:`_ringpolymermdKeys` section

   4. **qmmm-md** - Represents a hybrid quantum mechanics - molecular mechanics molecular dynamics simulation. (Not implemented yet)

   5. **mm-opt** - represents a geometry optimization calculation using molecular mechanics.

.. _timestepKey:

Timestep
========

.. admonition:: Key
    :class: tip

    timestep* = {double} fs

With the ``timestep`` keyword the time step in ``fs`` of one molecular dynamics loop is set.

.. _nstepKey:

NStep
=====

.. admonition:: Key
    :class: tip

    nstep* = {uint+}

The ``nstep`` keyword sets the total number of MD steps to be performed within this simulation run.

.. _floatingpointtypeKey:

Floating Point Type
===================

.. admonition:: Key
    :class: tip

    floating_point_type = {string} -> "double"

With the ``floating_point_type`` keyword the user can choose the precision of the floating point numbers used in the QM calculations if enabled by the external QM program/model.

Possible options are:

   1. **double** (default) - double precision floating point numbers are used

   2. **float** - single precision floating point numbers are used

.. _integratorKey:

Integrator
==========

.. admonition:: Key
    :class: tip

    integrator = {string} -> "v-verlet"

With the ``integrator`` keyword the user can choose the integrator type which should be applied.

Possible options are:

   1. **v-verlet** (default) - represents the Velocity-Verlet integrator 

.. _virialKey:

Virial
======

.. admonition:: Key
    :class: tip

    virial = {string} -> "molecular"

With the ``virial`` keyword the user can control if an intramolecular virial correction should be applied based on the definition of molecular units in the :ref:`moldescriptorFile` setup file.

Possible options are:

   1. **molecular** (default) - an intramolecular correction will be applied to the resulting virial from the force contributions

   2. **atomic** - no intramolecular correction to the resulting virial will be applied

.. _startfileKey:

Start File
==========

.. admonition:: Key
    :class: tip

    start_file* = {file}

The ``start_file`` keyword sets the name of the :ref:`restartFile` file for an MD simulation of any kind.

.. _rpmdstartfileKey:

RPMD Start File
===============

.. admonition:: Key
    :class: tip

    rpmd_start_file = {file}

The ``rpmd_start_file`` keyword is used to continue a ring polymer MD simulation containing positions, velocities and forces of all atoms of each bead of the ring polymer.

.. _generaloutputKeys:

General Output Keys
===================

.. Note::
    The **PQ** application has a special naming convention for output files. For every job type a certain set of :ref:`outputFiles` is written per default. If no output file names are given all prefixes of the output files will be named ``default.<ext>``. If at least one of the output file keys was given in the input file - the program will determine the most common prefix (*i.e.* string before the first ``.`` character) and set it for all unspecified output files.

    This feature enables easier post-processing of data and deletion of output files as **PQ** does not overwrite any existing output files.

.. _outputfreqKey:

Output Frequency
================

.. admonition:: Key
    :class: tip

    output_freq = {uint} -> 1

The ``output_freq`` keyword sets the frequency (*i.e.* every n-th step) of how often the application should write into the :ref:`outputFiles`. For a complete dry run without any output files the output frequency can be set to ``0``.

.. centered:: *default value* = 1

.. _fileprefixkey:

File Prefix
===========

.. admonition:: Key
    :class: tip

    file_prefix = {string}

The ``file_prefix`` keyword allows the user to set a common prefix name for all generated :ref:`outputFiles`.

.. _logfilekey:

Log File
===========

.. admonition:: Key
    :class: tip

    output_file = {file} -> "default.log"

The ``output_file`` keyword sets the name for the :ref:`logFile`, in which important information about the performed calculation can be found. 

.. centered:: *default value* = "default.log"

.. _mdoutputfilekeys:

*******************
MD Output File Keys
*******************

All of the following output files are written during calculations using MD related jobtypes.

.. _boxfilekey:

Box File
========

.. admonition:: Key
    :class: tip

    box_file = {file} -> "default.box"

The ``box_file`` keyword sets the name for the :ref:`boxFile`, which stores information about the simulation box throughout the simulation.

.. centered:: *default value* = "default.box"

.. _chargefilekey:

Charge File
===========

.. admonition:: Key
    :class: tip

    charge_file = {file} -> "default.chrg"

The ``charge_file`` keyword sets the name for the :ref:`chargeFile`, which stores the atomic partial charges throughout the MD simulation.

.. centered:: *default value* = "default.chrg"

.. _energyfilekey:

Energy File
===========

.. admonition:: Key
    :class: tip

    energy_file = {file} -> "default.en"

The ``energy_file`` keyword sets the name for the :ref:`energyFile`, in which the most important physical properties of the full simulation can be found.

.. centered:: *default value* = "default.en"

.. _forcefilekey:

Force File
==========

.. admonition:: Key
    :class: tip

    force_file = {file} -> "default.force"

The ``force_file`` keyword sets the name for the :ref:`forceFile`, which stores the atomic forces throughout the MD simulation.

.. centered:: *default value* = "default.force"

.. _infofilekey:

Info File
=========

.. admonition:: Key
    :class: tip

    info_file = {file} -> "default.info"

The ``info_file`` keyword sets the name for the :ref:`infoFile`, in which the most important physical properties of the last written step can be found.

.. centered:: *default value* = "default.info"

.. _instantenergyfilekey:

Instant Energy File
===================

.. admonition:: Key
    :class: tip

    instant_energy_file = {file} -> "default.instant_en"

The ``instant_energy_file`` keyword sets the name for the :ref:`instantEnergyFile`, in which the most important physical properties of the full simulation can be found.

.. centered:: *default value* = "default.instant_en"

.. _momentumfilekey:

Momentum File
=============

.. admonition:: Key
   :class: tip

    momentum_file = {file} -> "default.mom"

The ``momentum_file`` keyword sets the name for the :ref:`momentumFile`, which stores information about the total linear and total angular momentum throughout the MD simulation.

.. centered:: *default value* = "default.mom"

.. _referencefilekey:

Reference File
===============

.. admonition:: Key
   :class: tip

    reference_file = {file} -> "default.ref"

The ``reference_file`` keyword sets the name for the :ref:`refFile`, which lists the sources to be cited when publishing data with the chosen settings.

.. centered:: *default value* = "default.ref"

.. _rstfilekey:

Restart File
============

.. admonition:: Key
    :class: tip

    rst_file = {file} -> "default.rst"

The ``rst_file`` keyword sets the name for the :ref:`restartFile`, which contains all necessary information to restart (*i.e.* continue) the calculation from its current timestamp.

.. centered:: *default value* = "default.rst"

.. _stressfilekey:

Stress File
===========

.. admonition:: Key
    :class: tip

    stress_file = {file} -> "default.stress"

The ``stress_file`` keyword sets the name for the :ref:`stressFile`, which stores information about the stress tensor throughout the MD simulation.

.. centered:: *default value* = "default.stress"

.. _timingsfilekey:

Timings File
============

.. admonition:: Key
    :class: tip

    timings_file = {file} -> "default.timings"

The ``timings_file`` keyword sets the name for the :ref:`timingFile`, which tracks the time **PQ** takes for executing individual parts of the simulation.

.. centered:: *default value* = "default.timings"

.. _trajectoryfilekey:

Trajectory File
===============

.. admonition:: Key
    :class: tip

    traj_file = {file} -> "default.xyz"

The ``traj_file`` keyword sets the name for the :ref:`trajectoryFile`, which stores the atomic positions throughout the MD simulation.

.. centered:: *default value* = "default.xyz"

.. _velocityfilekey:

Velocity File
=============

.. admonition:: Key
    :class: tip

    vel_file = {file} -> "default.vel"

The ``vel_file`` keyword sets the name for the :ref:`velocityFile`, which stores the atomic velocities throughout the MD simulation.

.. centered:: *default value* = "default.vel"

.. _virialfilekey:

Virial File
===========

.. admonition:: Key
    :class: tip

    virial_file = {file} -> "default.vir"

The ``virial_file`` keyword sets the name for the :ref:`virialFile`, which stores information about the virial tensor throughout the MD simulation.

.. centered:: *default value* = "default.vir"

.. _rpmdoutputfilekeys:

*********************
RPMD Output File Keys
*********************

All of the following output files are written during calculations using ring polymer MD jobtype. The files represent the trajectories of all individual beads.

.. _rpmdchargefilekey:

RPMD Charge File
================

.. admonition:: Key
    :class: tip

    rpmd_charge_file = {file} -> "default.rpmd.chrg"

The ``rpmd_charge_file`` keyword sets the name for the file containing partial charges of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.chrg"

.. _rpmdenergyfilekey:

RPMD Energy File
================

.. admonition:: Key
    :class: tip

    rpmd_energy_file = {file} -> "default.rpmd.en"

The ``rpmd_energy_file`` keyword sets the name for the file containing relevant energy data for each ring polymer bead of the simulation.

.. centered:: *default value* = "default.rpmd.en"

.. _rpmdforcefilekey:

RPMD Force File
===============

.. admonition:: Key
    :class: tip

    rpmd_force_file = {file} -> "default.rpmd.force"

The ``rpmd_force_file`` keyword sets the name for the file containing forces of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.force"


.. _rpmdrestartfilekey:

RPMD Restart File
=================

.. admonition:: Key
    :class: tip

    rpmd_restart_file = {file} -> "default.rpmd.rst"

The ``rpmd_restart_file`` keyword sets the name for the ring polymer restart file, which contains all necessary information to restart (*i.e.* continue) the calculation from its current timestamp.

.. centered:: *default value* = "default.rpmd.rst"

.. _rpmdtrajectoryfilekey:

RPMD Trajectory File
====================

.. admonition:: Key
    :class: tip

    rpmd_traj_file = {file} -> "default.rpmd.xyz"

The ``rpmd_traj_file`` keyword sets the name for the file containing positions of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.xyz"

.. _rpmdvelocityfilekey:

RPMD Velocity File
==================

.. admonition:: Key
    :class: tip

    rpmd_vel_file = {file} -> "default.rpmd.vel"

The ``rpmd_vel_file`` keyword sets the name for the file containing velocities of all atoms of each bead of the ring polymer trajectory.

.. centered:: *default value* = "default.rpmd.vel"

.. _setupfilekeys:

****************
Setup File Keys
****************

In order to set up certain calculations additional input files have to be used. The names of these setup files have to be specified in the 
input file. Further information about the individual files can be found in the :ref:`setupFiles` section.

.. _moldescriptorfileKey:

Moldesctiptor File
==================

.. admonition:: Key
    :class: tip

    moldescriptor_file = {file} -> "moldescriptor.dat"

.. _gufffileKey:

GUFF File
=========

.. admonition:: Key
    :class: tip

    guff_file = {file} -> "guff.dat"

.. _dftbfileKey:

DFTB Setup File
===============

.. admonition:: Key
    :class: tip

    dftb_file = {file} -> "dftb_in.template"

.. _topologyFileKey:

Topology File
=============

.. admonition:: Key
    :class: tip

    topology_file = {file}

.. _parameterFileKey:

Parameter File
==============

.. admonition:: Key
    :class: tip

    parameter_file = {file}

MSHake_File
===========

.. admonition:: Key
    :class: tip

    mshake_file = {file}

.. _intraNonBondedFileKey:

Intra-NonBonded_File
====================

.. admonition:: Key
    :class: tip

    intra-nonbonded_file = {file}

.. _simulationboxKeys:

*******************
Simulation Box Keys
*******************

.. _densityKey:

Density
=======

.. admonition:: Key
    :class: tip

    density = {double} kgL⁻¹

With the ``density`` keyword the box dimension of the system can be inferred from the total mass of the simulation box.

.. Note::
    This keyword implies that the simulation box has a cubic shape. Furthermore, the ``density`` keyword will be ignored if in the :ref:`restartFile` contains any box information.

.. _radialCoulombCutoffKey:

Radial Coulomb Cutoff
=====================

.. admonition:: Key
    :class: tip


    rcoulomb = {double} :math:`\mathrm{\mathring{A}}` -> 12.5 :math:`\mathrm{\mathring{A}}`

With the ``rcoulomb`` keyword the radial cut-off in :math:`\mathrm{\mathring{A}}` of Coulomb interactions for MM-MD type simulations can be set. If pure QM-MD type simulations are applied this keyword will be ignored and the value will be set to 0 :math:`\mathrm{\mathring{A}}`.

.. centered:: *default value* = 12.5 :math:`\mathrm{\mathring{A}}` (for MM-MD type simulations)

.. _initialvelocitiesKey:

Initial Velocities
==================

.. admonition:: Key
    :class: tip

    init_velocities = {bool} -> false

To initialize the velocities of the system according to the target temperature with a Boltzmann distribution the user has to set the ``init_velocities`` to true.

Possible options are:

   1. **false** (default) - velocities are taken from start file

   2. **true** - velocities are initialized according to a Boltzmann distribution at the target temperature.

.. _temperatureCouplingKeys:

*************************
Temperature Coupling Keys
*************************

.. _temperatureKey:
 
Temperature
===========

.. admonition:: Key
    :class: tip

    temp = {double} K

With the ``temp`` keyword the target temperature in ``K`` of the system can be set. 

.. Note::
    This keyword is not restricted to the use of any temperature coupling method, as it is used *e.g.* also for the initialization of Boltzmann distributed velocities or the reset of the system temperature.

.. _startingTemperatureKey:

Starting Temperature
====================

.. admonition:: Key
    :class: tip

    start_temp = {double} K

With the ``start_temp`` keyword the initial temperature in ``K`` of the system can be set. If a value is given the PQ application will perform a temperature ramping from the ``start_temp`` to the ``temp`` value.

.. _endTemperatureKey:

End Temperature
===============

.. admonition:: Key
    :class: tip

    end_temp = {double} K

The ``end_temp`` keyword is a synonym for the ``temp`` keyword and can be used to set the target temperature of the system. It cannot be used in combination with the ``temp`` keyword.

.. _temperaturerampstepsKey:

Temperature Ramp Steps
======================

.. admonition:: Key
    :class: tip

    temp_ramp_steps = {uint+}

With the ``temp_ramp_steps`` keyword the user can specify the number of steps for the temperature ramping from the ``start_temp`` to the ``temp`` value. If no starting temperature is given the keyword will be ignored. If a starting temperature is given and this keyword is omitted the temperature ramping will be performed over the full simulation time.

.. centered:: *default value* = full simulation time

.. _temperatureRampFrequencyKey:

Temperature Ramp Frequency
==========================

.. admonition:: Key
    :class: tip

    temp_ramp_frequency = {uint+} -> 1

With the ``temp_ramp_frequency`` keyword the user can specify the frequency of the temperature ramping from the ``start_temp`` to the ``temp`` value. If no starting temperature is given the keyword will be ignored. If a starting temperature is given and this keyword is omitted the temperature ramping will be performed, so that each step the temperature is increased by the same value.

.. centered:: *default value* = 1 step

.. _thermostatKey:

Thermostat
==========
.. TODO: reference manual

.. admonition:: Key
    :class: tip

    thermostat = {string} -> "none"

With the ``thermostat`` keyword the temperature coupling method can be chosen.

Possible options are:

   1. **none** (default) - no thermostat is set, hence {N/µ}{p/V}E settings are applied.

   2. **berendsen** - the `Berendsen <https://doi.org/10.1063/1.448118>`_ weak coupling thermostat. Based on the rescaling of velocities according to the scaling factor :math:`\zeta`, equation :eq:`BerendsenThermostatEquation`. Ideal for crude temperature adjustments. Not able to reproduce the correct canonical ensemble.

        .. math:: \zeta = \sqrt{1 + \frac{\Delta t}{\tau} \left( \frac{T_0}{T} - 1 \right)}
            :label: BerendsenThermostatEquation

   3. **velocity_rescaling** - the stochastic velocity rescaling thermostat also known as `Bussi-Donadio-Parrinello <https://doi.org/10.1063/1.2408420>`_ thermostat. Based on the rescaling of velocities according to the scaling factor :math:`\zeta`, equation :eq:`BussiDonadioParrinelloThermostatEquation`. Enforces a canonical kinetic energy distribution.

        .. math:: \zeta = \sqrt{1 + \frac{\Delta t}{\tau} \left( \frac{T_0}{T} - 1 +2 \sqrt{\frac{T_0}{T} \frac{\Delta t}{\tau} \frac{1}{df}} dW \right)}
            :label: BussiDonadioParrinelloThermostatEquation

   4. **langevin** - temperature coupling *via* stochastic Langevin dynamics. Based on modifying the force of each individual particle :math:`F_{\text i}` *via* a friction force :math:`\gamma \cdot p_{\text i}` and a random force :math:`\xi`, equation :eq:`LangevinThermostatEquation`. The friction coefficient :math:`\gamma` can be set with the :ref:`frictionKey` keyword. Enforces a canonical kinetic energy distribution. However, the Langevin thermostat is unable to conserve the total momentum of the system, which may lead to critical erros in the resulting dynamical data.

        .. math:: m_{\text i} \dot{v}_{\text i} = F_{\text i} - \gamma \cdot p_{\text i} + \xi
            :label: LangevinThermostatEquation

   5. **nh-chain** - temperature coupling *via* `Nose Hoover extended Lagrangian <https://doi.org/10.1063/1.463940>`_. Based on modifying the forces after each time step. The length of the Nose Hoover chain and the coupling frequency can be set with the :ref:`nhchainlenghtKey` and the :ref:`couplingFrequencyKey` keywords, respectively. Enforces a canonical kinetic energy distribution.

.. _temperatureRelaxationTimeKey:

Temperature Relaxation Time
===========================

This keyword is used in combination with the Berendsen and velocity rescaling thermostat.

.. admonition:: Key
    :class: tip

    t_relaxation = {double} ps -> 0.1 ps

With the ``t_relaxation`` keyword the relaxation time in ``ps`` (*i.e.* :math:`\tau`) of the Berendsen or stochastic velocity rescaling thermostat is set, see equations :eq:`BerendsenThermostatEquation` and :eq:`BussiDonadioParrinelloThermostatEquation`.

.. centered:: *default value* = 0.1 ps

.. _frictionKey:

Friction
========

.. admonition:: Key
    :class: tip

    friction = {double} ps⁻¹ -> 0.1 ps⁻¹

With the ``friction`` keyword the friction coefficient :math:`\gamma` in ``ps⁻¹`` of the Langevin thermostat, equation :eq:`LangevinThermostatEquation`, can be set.

.. centered:: *default value* = 0.1 ps⁻¹

.. _nhchainlenghtKey:

NH-Chain Length
===============

.. admonition:: Key
    :class: tip

    nh-chain_length = {uint+} -> 3

With the ``nh-chain_length`` keyword the length of the chain for temperature control *via* an extended Nose-Hoover Lagrangian can be set.

.. centered:: *default value* = 3

.. _couplingFrequencyKey:

Coupling Frequency
==================

.. admonition:: Key
    :class: tip

    coupling_frequency = {double} cm⁻¹ -> 1000 cm⁻¹

With the ``coupling_frequency`` keyword the coupling frequency of the Nose-Hoover chain in ``cm⁻¹`` can be set.

.. centered:: *default value* = 1000 cm⁻¹

.. _pressureCouplingKeys:

**********************
Pressure Coupling Keys
**********************

.. _pressureKey:

Pressure
========

.. admonition:: Key
    :class: tip

    pressure = {double} bar

With the ``pressure`` keyword the target pressure in ``bar`` of the system can be set. 

.. Note::
    This keyword is only used if a manostat for controlling the pressure is explicitly defined.

.. _manostatKey:

Manostat
========
.. TODO: reference manual

.. admonition:: Key
    :class: tip

    manostat = {string} -> "none"

With the ``manostat`` keyword the type of pressure coupling can be chosen.

Possible options are:

   1. **none** (default) - no pressure coupling is applied (*i.e.* constant volume)

   2. **berendsen** - Berendsen weak coupling manostat

   3. **stochastic_rescaling** - stochastic cell rescaling manostat

.. _pressureRelaxationKey:

Pressure Relaxation
===================

This keyword is used in combination with the Berendsen and stochastic cell rescaling manostat.

.. admonition:: Key
    :class: tip

    p_relaxation = {double} ps -> 0.1 ps

With the ``p_relaxation`` keyword the relaxation time in ``ps`` (*i.e.* :math:`\tau`) of the Berendsen or stochastic cell rescaling manostat is set.

.. centered:: *default value* = 0.1 ps

.. _compressibilityKey:

Compressibility
===============

This keyword is used in combination with the Berendsen and stochastic cell rescaling manostat.

.. admonition:: Key
    :class: tip

    compressibility = {double} bar⁻¹ -> 4.591e-5 bar⁻¹

With the ``compressibility`` keyword the user can specify the compressibility of the target system in ``bar⁻¹`` for the Berendsen and stochastic cell rescaling manostat.

.. centered:: *default value* = 4.591e-5 bar⁻¹ (compressibility of water)

.. _isotropyKey:

Isotropy
========

.. admonition:: Key
    :class: tip

    isotropy = {string} -> "isotropic"

With the ``isotropy`` keyword the isotropy of the pressure coupling for all manostat types is controlled.

Possible options are:

   1. **isotropic** (default) - all axes are scaled with the same scaling factor

   2. **xy** - semi-isotropic settings, with axes ``x`` and ``y`` coupled isotropic

   3. **xz** - semi-isotropic settings, with axes ``x`` and ``z`` coupled isotropic

   4. **yz** - semi-isotropic settings, with axes ``y`` and ``z`` coupled isotropic

   5. **anisotropic** - all axes are coupled in an anisotropic way

   6. **full_anisotropic** - all axes are coupled in an anisotropic way and the box angles are also scaled

.. _resetKineticsKeys:

*******************
Reset Kinetics Keys
*******************

.. _nscaleKey:

NScale
======

.. admonition:: Key
    :class: tip

    nscale = {uint} -> 0

With the ``nscale`` keyword the user can specify the first ``n`` steps in which the temperature is reset *via* a hard scaling approach to the target temperature.

.. Note::
    Resetting the temperature to the target temperature does imply also a subsequent reset of the total box momentum. Furthermore, resetting to the target temperature does not necessarily require a constant temperature ensemble setting.

.. centered:: *default value* = 0 (*i.e.* never)

.. _fscaleKey:

FScale
======

.. admonition:: Key
    :class: tip

    fscale = {uint} -> nstep + 1

With the ``fscale`` keyword the user can specify the frequency ``f`` at which the temperature is reset *via* a hard scaling approach to the target temperature.

.. Note:: 
    Resetting the temperature to the target temperature does imply also a subsequent reset of the total box momentum. Furthermore, resetting to the target temperature does not necessarily require a constant temperature ensemble setting.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1 

.. _nresetKey:

NReset
======

.. admonition:: Key
    :class: tip

    nreset = {uint} -> 0

With the ``nreset`` keyword the user can specify the first ``n`` steps in which the total box momentum is reset.

.. centered:: *default value* = 0 (*i.e.* never)

.. _fresetKey:

FReset
======

.. admonition:: Key
    :class: tip

    freset = {uint} -> nstep + 1

With the ``freset`` keyword the user can specify the frequency ``f`` at which the total box momentum is reset.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1

.. _nresetangularKey:

NReset Angular
==============

.. admonition:: Key
    :class: tip

    nreset_angular = {uint} -> 0

With the ``nreset_angular`` keyword the user can specify the first ``n`` steps in which the total angular box momentum is reset.

.. Danger::
    This setting should be used very carefully, since in periodic systems a reset of the angular momentum can result in severe unphysical behavior.

.. centered:: *default value* = 0 (*i.e.* never)

.. _freseangularKey:

FReset Angular
==============

.. admonition:: Key
    :class: tip

    freset_angular = {uint} -> nstep + 1

With the ``freset_angular`` keyword the user can specify the frequency ``f`` at which the total angular box momentum is reset.

.. Danger::
    This setting should be used very carefully, since in periodic systems a reset of the angular momentum can result in severe unphysical behavior.

.. centered:: *default value* = nstep + 1 (*i.e.* never)

.. centered:: *special case* = 0 -> nstep + 1 

.. _constraintsKeys:

****************
Constraints Keys
****************

.. _shakeKey:

Shake
=====

.. admonition:: Key
    :class: tip

    shake = {string} -> "off"

With the ``shake`` keyword the SHAKE/RATTLE algorithm for bond constraints can be activated.

Possible options are:

   1. **off** (default) - no shake will be applied

   2. **on** - SHAKE for bond constraints defined in the :ref:`topologyFile` will be applied.

   3. **shake** - SHAKE for bond constraints defined in the :ref:`topologyFile` will be applied.

   4. **mshake** - M-SHAKE for bond constraints defined in a special :ref:`mshakeFile` will be applied. As the M-SHAKE algorithm is designed for the treatment of rigid body molecular units the general shake algorithm will be activated automatically along with the M-SHAKE algorithm. The shake bonds can be defined as usual in the :ref:`topologyFile` and if no SHAKE bonds are defined only the M-SHAKE algorithm will be applied (without any overhead)

.. _shaketoleranceKey:

Shake Tolerance
===============

.. admonition:: Key
    :class: tip

    shake-tolerance = {double} -> 1e-8

With the ``shake-tolerance`` keyword the user can specify the tolerance, with which the bond length of the shaked bonds should converge.

.. centered:: *default value* = 1e-8

.. _shakeiterationKey:

Shake Iteration
===============

.. admonition:: Key
    :class: tip

    shake-iter = {uint+} -> 20

With the ``shake-iter`` keyword the user can specify the maximum number of iterations until the convergence of the bond lengths should be reached within the shake algorithm.

.. centered:: *default value* = 20

.. _rattletoleranceKey:

Rattle Tolerance
================

.. admonition:: Key
    :class: tip


    rattle-tolerance = {double} s⁻¹kg⁻¹ -> 1e4 s⁻¹kg⁻¹ 


With the ``rattle-tolerance`` keyword the user can specify the tolerance in ``s⁻¹kg⁻¹``, with which the velocities of the shaked bonds should converge.

.. centered:: *default value* = 20 s⁻¹kg⁻¹

.. _rattleiterationKey:

Rattle Iteration
================

.. admonition:: Key
    :class: tip

    rattle-iter = {uint+} -> 20

With the ``rattle-iter`` keyword the user can specify the maximum number of iterations until the convergence of the velocities of the shaked bonds should be reached within the rattle algorithm.

.. centered:: *default value* = 20

.. _distanceConstraintsKey:

Distance Constraints
====================

.. admonition:: Key
    :class: tip

    distance-constraints = {string} -> "off"

With the ``distance-constraints`` keyword it is possible to activate distance constraints for the simulation. The distance constraints are defined *via* the :ref:`topologyFile`.

.. _mmKeys:

*******
MM Keys
*******

.. _noncoulombKey:

NonCoulomb
==========

.. admonition:: Key
    :class: tip

    noncoulomb = {string} -> "guff"

With the ``noncoulomb`` keyword the user can specify which kind of GUFF formalism should be used for parsing the :ref:`guffdatFile`.

.. Note::

    This keyword is only considered if an MM-MD type simulation is requested and the :ref:`forcefieldKey` is turned off.

Possible options are:

   1. **guff** (default) - full GUFF formalism

   2. **lj** - Lennard Jones quick routine

   3. **buck** - Buckingham quick routine

   4. **morse** - Morse quick routine

.. _forcefieldKey:

Force Field
===========

.. admonition:: Key
    :class: tip

    forcefield = {string} -> "off"

With the ``forcefield`` keyword the user can switch from the GUFF formalism to a force field type simulation (For details see Reference Manual).

Possible options are:

   1. **off** (default) - GUFF formalism is applied

   2. **on** - full force field definition is applied

   3. **bonded** - non bonded interaction are described *via* GUFF formalism and bonded interactions *via* force field approach

.. _longrangecorrectionKeys:

*********************
Long Range Correction
*********************

.. _longrangeKey:

Long Range
==========

.. admonition:: Key
    :class: tip

    long_range = {string} -> "none"

With the ``long_range`` correction keyword the user can specify the type of Coulombic long range correction, which should be applied during the simulation.

Possible options are:

   1. **none** (default) - no long range correction

   2. **wolf** - Wolf summation

.. _wolfParameterKey:

Wolf Parameter
==============
.. TODO: add unit and description

.. admonition:: Key
    :class: tip

    wolf_param = {double} -> 0.25 

.. centered:: *default value* = 0.25

.. _qmKeys:

*******
QM Keys
*******

.. _qmprogamKey:

QM Program
==========

.. admonition:: Key
    :class: tip

    qm_prog = {string}

With the ``qm_prog`` keyword the external QM engine for any kind of QM MD simulation is chosen.

.. Note::
    This keyword is required for any kind of QM MD simulation!

Possible options are:

   1. **dftbplus** - `DFTB+ <https://dftbplus.org/index.html>`_

   2. **pyscf** - `PySCF <https://pyscf.org/>`_

   3. **turbomole** - `Turbomole <https://www.turbomole.org/>`_

   4. **mace**  - `MACE-MP <https://arxiv.org/abs/2401.00096>`_ same as using **mace_mp**

   5. **mace_off** - `MACE-OFF23 <https://arxiv.org/abs/2312.15211>`_

   6. **ase-dftbplus** - `DFTB+ <https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html#module-ase.calculators.dftb>`_ called by `ASE <https://wiki.fysik.dtu.dk/ase/>`_ 
   
.. _qmscriptKey:

QM Script
=========

.. admonition:: Key
    :class: tip

    qm_script = {file}

With the ``qm_script`` keyword the external executable to run the QM engine and to parse its output is chosen. All possible scripts can be found under `<https://github.com/MolarVerse/PQ/tree/main/src/QM/scripts>`_. Already the naming of the executables should hopefully be self-explanatory in order to choose the correct input executable name.

.. _qmscriptfullpathKey:

QM Script Full Path
===================

.. admonition:: Key
    :class: tip

    qm_script_full_path = {pathFile}

.. attention::
   This keyword can not be used in conjunction with the ``qm_script`` keyword! Furthermore, this keyword needs to be used in combination with any singularity or static build of PQ. For further details regarding the compilation/installation please refer to the :ref:`userG_installation` section.

With the ``qm_script_full_path`` keyword the user can specify the full path to the external executable to run the QM engine and to parse its output. All possible scripts can be found under `<https://github.com/MolarVerse/PQ/tree/main/src/QM/scripts>`_. Already the naming of the executables should hopefully be self-explanatory in order to choose the correct input executable name.

.. _qmlooptimelimitKey:

QM Loop Time Limit
==================

.. admonition:: Key
    :class: tip

    qm_loop_time_limit = {double} s -> -1 s

With the ``qm_loop_time_limit`` keyword the user can specify the loop time limit in ``s`` of all QM type calculations. If the time limit is reached the calculation will be stopped. Default value is -1 s, which means no time limit is set, and the calculation will continue until it is finished. In general all negative values will be interpreted as no time limit.

.. _disperstoncorrectionKey:

Dispersion Correction
=====================

.. admonition:: Key
    :class: tip

    dispersion = {bool} -> false

With the ``dispersion`` keyword the user can activate the dispersion correction for the QM calculations - at the moment only enabled for ASE based QM engines.

.. _maceModelSizeKey:

MACE Model Size
===============

.. admonition:: Key
    :class: tip

    mace_model_size = {string} -> "medium"

With the ``mace_model_size`` keyword the user can specify the size of the `MACE <https://arxiv.org/abs/2206.07697>`_ model for the QM calculations.

Possible options are:

   1. **small** - small MACE model

   2. **medium** (default) - medium MACE model

   3. **large** - large MACE model


.. _slakosTypeKey:

ASE-DFTB+ Approach
=============

.. admonition:: Key
    :class: tip

    slakos = {string}

With the ``slakos`` keyword the user can specify the type of the ``ase-dftbplus`` approach for DFTB+ calculations.

Possible options are:

    1. **3ob** - 3ob parameters

    2. **matsci** - matsci parameters

    3. **custom** - custom parameters (see :ref:`slakosPathKey`)

.. _slakosPathKey:

Slater-Koster Path
==================

.. admonition:: Key
    :class: tip

    slakos_path = {pathFile}

With the ``slakos_path`` keyword the user can specify the path to the Slater-Koster files for ASE DFTB+ calculations. Is only compatible, but mandatory if the ``slakos`` keyword is set to **custom**.

.. _thirdOrderKey:

Third-Order
===========

.. admonition:: Key
    :class: tip

    third_order = {bool} -> false

With the ``third_order`` keyword the user can activate the 3rd order DFTB expansion according to `Grimme et al. <https://pubs.acs.org/doi/10.1021/ct100684s>`_ for ASE DFTB+ calculations. Is automatically set to ``true`` if the ``slakos`` keyword is set to **3ob**.

.. _hubbardDerivKey:

Hubbard Derivatives
===================

.. admonition:: Key
    :class: tip

    hubbard_derivs = {dict}

If the Slater-Koster parameters are of DFTB3 type, the Hubbard derivatives can be set with the ``hubbard_derivs`` keyword. The Hubbard derivatives are given as a dictionary with the chemical element as key and the Hubbard derivative as value. Standard Hubbard derivatives for ``slakos`` **3ob** are preset according to `dftb.org <https://github.com/dftbparams/3ob>`_ and can be overwritten by the user.

.. admonition:: Example
    :class: code

    hubbard_derivs = C: 0.1, H: 0.2, O: 0.3;




.. _ringpolymermdKeys:

********************
Ring Polymer MD Keys
********************

.. _rpmdnreplicaKey:

RPMD n replica
==============

.. admonition:: Key
    :class: tip

    rpmd_n_replica = {uint+}

With the ``rpmd_n_replica`` keyword the number of beads for a ring polymer MD simulation is controlled.

.. Note::
    This keyword is required for any kind of ring polymer MD simulation!

.. _qmmmKeys:

**********
QM/MM Keys
**********

.. _qmcenterKey:

QM Center
=========

.. admonition:: Key
    :class: tip

    qm_center = {selection} -> 0

With the ``qm_center`` keyword the user can specify the center of the QM region. The default selection is the first atom of the system (*i.e.* 0). For more information about the selection grammar see the `selectionType`_ section. The ``qm_center`` if more than one atom is selected will be by default the center of mass of the selected atoms.

.. _qmonlylistKey:

QM Only List
============

.. admonition:: Key
    :class: tip

    qm_only_list = {selection}

With the ``qm_only_list`` keyword the user can specify a list of atoms which should be treated as QM atoms only. This means that these atoms can not leave the QM region during the simulation. For more information see the reference manual. For more information about the selection grammar see the `selectionType`_ section. By default no atoms are selected.

.. _mmonlylistKey:

MM Only List
============

.. admonition:: Key
    :class: tip

    mm_only_list = {selection}

With the ``mm_only_list`` keyword the user can specify a list of atoms which should be treated as MM atoms only. This means that these atoms can not enter the QM region during the simulation. For more information see the reference manual. For more information about the selection grammar see the `selectionType`_ section. By default no atoms are selected.

.. _qmchargesKey:

QM Charges
==========

.. admonition:: Key
    :class: tip

    qm_charges = {string} -> "off"

With the ``qm_charges`` keyword the user can specify the charge model for the QM atoms.

Possible options are:

   1. **off** (default) - charges of the QM atoms are taken from the MM model

   2. **on** - charges of the QM atoms are taken from the QM calculation

.. _qmcoreradiusKey:

QM Core Radius
==============

.. admonition:: Key
    :class: tip

    qm_core_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}}`

With the ``qm_core_radius`` keyword the user can specify the core radius in :math:`\mathrm{\mathring{A}}` around the ``qm_center``. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that the core radius is not set and only explicit QM atoms are used for the QM region.

.. _qmmmlayerradiuskey:

QM/MM Layer Radius
==================

.. admonition:: Key
    :class: tip

    qmmm_layer_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}`

With the ``qmmm_layer_radius`` keyword the user can specify the layer radius in :math:`\mathrm{\mathring{A}}` around the ``qm_center``. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that no special QM/MM treatment is applied.

.. _qmmmsmoothingradiuskey:

QM/MM Smoothing Radius
======================

.. admonition:: Key
    :class: tip

    qmmm_smoothing_radius = {double} :math:`\mathrm{\mathring{A}}` -> 0.0 :math:`\mathrm{\mathring{A}`

With the ``qmmm_smoothing_radius`` keyword the user can specify the smoothing radius in :math:`\mathrm{\mathring{A}}` of the QM atoms. The default value is 0.0 :math:`\mathrm{\mathring{A}}`, which means that the smoothing radius is not set and no smoothing is applied.

.. _celllistKeys:

**************
Cell List Keys
**************

.. _celllistKey:

Cell List
=========

.. admonition:: Key
    :class: tip

    cell-list = {string} -> "off"

With the ``cell-list`` the user can activate a cell-list approach to calculate the pair-interactions in MM-MD simulations (no effect in pure QM-MD type simulations).

Possible options are:

   1. **off** (default) - brute force routine

   2. **on** - cell list approach is applied

.. _cellnumberKey:

Cell Number
===========

.. admonition:: Key
    :class: tip

    cell-number = {uint+} -> 7

With the ``cell-number`` keyword the user can set the number of cells in each direction in which the simulation box will be split up (*e.g.* cell-number = 7 -> total cells = 7x7x7)

.. centered:: *default value* = 7

.. _optimizationKeys:

*****************
Optimization Keys
*****************

In order to perform a geometry optimization one of the optimizer :ref:`Jobtypes <jobtype>` has to be chosen.

.. _optimizerKey:

Optimizer
=========

.. admonition:: Key
    :class: tip

    optimizer = {string}

This keyword is mandatory for any kind of geometry optimization. The user has to specify the optimizer which should be used for the optimization.

Possible options are:

   1. **steepest-descent** - steepest descent optimizer

   2. **ADAM** - ADAM optimizer

.. _learningratestrategyKey:

Learning Rate Strategy
======================

.. admonition:: Key
    :class: tip

    learning-rate-strategy = {string} -> "exponential-decay"

With the ``learning-rate-strategy`` keyword the user can specify the learning rate strategy for all kind of optimization jobs.

Possible options are:

   1. **exponential-decay** (default) - exponential decay of the learning rate

   2. **constant** - constant learning rate

   3. **constant-decay** - constant decay of the learning rate

.. _initiallearningrateKey:

Initial Learning Rate
=====================

.. admonition:: Key
    :class: tip

    initial-learning-rate = {double} -> 0.0001

With the ``initial-learning-rate`` keyword the user can specify the initial learning rate for all kind of optimization jobs.

.. centered:: *default value* = 0.0001

.. _learningratedecayKey:

Learning Rate Decay
===================

.. admonition:: Key
    :class: tip

    learning-rate-decay = {double}

With the ``learning-rate-decay`` keyword the user can specify the decay speed of the learning rate. Pay attention this key is used at the moment for different kind of decay strategies and therefore the value is dependent on the chosen strategy.

.. _convergenceKeys:

****************
Convergence Keys
****************

.. _energyconvergencestrategyKey:

Energy Convergence Strategy
===========================

In general the convergence of the geometry optimization is checked by assuring that the absolute **and** relative energy difference between two consecutive steps is smaller than a certain threshold. The user can choose between different strategies to change this behavior.

.. admonition:: Key
    :class: tip

    energy-conv-strategy = {string} -> "rigorous"

With the ``energy-conv-strategy`` keyword the user can specify the energy convergence strategy for all kind of optimization jobs.

Possible options are:

   1. **rigorous** (default) - both absolute and relative energy difference have to be smaller than the threshold

   2. **loose** - only one of the two energy differences has to be smaller than the threshold

   3. **relative** - only the relative energy difference has to be smaller than the threshold

   4. **absolute** - only the absolute energy difference has to be smaller than the threshold

.. _energyconvergencecheckKey:

Enable/Disable Energy Convergence Check
=======================================

.. admonition:: Key
    :class: tip

    use-energy-conv = {bool} -> true

With the ``use-energy-conv`` keyword the user can enable or disable the energy convergence check for all kind of optimization jobs.

.. _maxforceconvergencecheckKey:

Enable/Disable MAX Force Convergence Check
==========================================

.. admonition:: Key
    :class: tip

    use-max-force-conv = {bool} -> true

With the ``use-max-force-conv`` keyword the user can enable or disable the maximum force convergence check for all kind of optimization jobs.

.. _rmsforceconvergencecheckKey:

Enable/Disable RMS Force Convergence Check
==========================================

.. admonition:: Key
    :class: tip

    use-rms-force-conv = {bool} -> true

With the ``use-rms-force-conv`` keyword the user can enable or disable the root mean square force convergence check for all kind of optimization jobs.

.. _energyconvergencethresholdKey:

Energy Convergence Threshold
============================

.. admonition:: Key
    :class: tip

    energy-conv = {double} -> 1e-6

With the ``energy-conv`` keyword the user can specify the energy convergence threshold for all kind of optimization jobs.
This keyword will set both the absolute and relative energy convergence threshold.

.. centered:: *default value* = 1e-6

.. _relativeenergyconvergencethresholdKey:

Relative Energy Convergence Threshold
=====================================

.. admonition:: Key
    :class: tip

    rel-energy-conv = {double} -> 1e-6

With the ``rel-energy-conv`` keyword the user can specify the relative energy convergence threshold for all kind of optimization jobs. This keyword overrides the ``energy-conv`` keyword.

.. centered:: *default value* = 1e-6

.. _absoluteenergyconvergencethresholdKey:

Absolute Energy Convergence Threshold
=====================================

.. admonition:: Key
    :class: tip

    abs-energy-conv = {double} -> 1e-6

With the ``abs-energy-conv`` keyword the user can specify the absolute energy convergence threshold for all kind of optimization jobs. This keyword overrides the ``energy-conv`` keyword.

.. centered:: *default value* = 1e-6

.. _forceconvergencethresholdKeys:

Force Convergence Threshold
===========================

.. admonition:: Key
    :class: tip

    force-conv = {double} -> 1e-6

With the ``force-conv`` keyword the user can specify the force convergence threshold for all kind of optimization jobs. This keyword will set both the maximum and root mean square force convergence threshold.

.. centered:: *default value* = 1e-6

.. _maxforceconvergencethresholdKey:

Maximum Force Convergence Threshold
===================================

.. admonition:: Key
    :class: tip

    max-force-conv = {double} -> 1e-6

With the ``max-force-conv`` keyword the user can specify the maximum force convergence threshold for all kind of optimization jobs. This keyword overrides the ``force-conv`` keyword.

.. centered:: *default value* = 1e-6

.. _rmsforceconvergencethresholdKey:

RMS Force Convergence Threshold
===============================

.. admonition:: Key
    :class: tip

    rms-force-conv = {double} -> 1e-6

With the ``rms-force-conv`` keyword the user can specify the root mean square force convergence threshold for all kind of optimization jobs. This keyword overrides the ``force-conv`` keyword.

.. centered:: *default value* = 1e-6