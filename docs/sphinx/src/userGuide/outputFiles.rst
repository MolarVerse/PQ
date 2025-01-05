.. _outputFiles: 

############
Output Files
############

The following output files are printed by **PQ**. The individual file names can be changed *via* the corresponding 
:ref:`mdoutputfilekeys` in the ``.in`` file. Alternatively, a common prefix for all output files can be provided using the
:ref:`file_prefix <fileprefixkey>` key. 

All data blocks shown for a single simulation step or frame are concatenated without blank lines in the output files.

.. _boxFile:

*********
Box File
*********

**File Type:** ``.box``

Contains the three lattice parameters (*a*, *b*, *c*) and the three angles (*α*, *β*, *γ*) of the simulation box for 
every frame in the following format:
    
    step_number *a* *b* *c* *α* *β* *γ*

The lattice parameters are given in units of Å and the angles are given in units of degrees.

.. _chargeFile:

************
Charge File
************

**File Type:** ``.chrg``

Stores the charge of each atom for every frame of the simulation in the following format:
    
    | **line 1:** n_atoms *a* *b* *c* *α* *β* *γ*
    | **line 2:** empty
    | **line 3 to (n_atoms + 2):** atom_type_name charge

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters in units of Å and degrees. The second line is left empty. The following lines contain the name 
of the atom type (as given in the :ref:`moldescriptorFile` file) and its charge in units of the elementary charge *e* for each atom in the system.

.. _energyFile:

***********
Energy File
***********

**File Type:** ``.en``

Stores information about the energy and various other quantities of the system for every frame in the following format:

    step_number *T* *P* *E*:sub:`tot` *E*:sub:`QM` *N*:sub:`QM-atoms` *E*:sub:`kin` *E*:sub:`intra` *E*:sub:`Coulomb` *E*:sub:`non-Coulomb` *E*:sub:`bond` *E*:sub:`angle` *E*:sub:`dihedral` *E*:sub:`improper` *V* *ρ* *E*:sub:`p-NH` *E*:sub:`χ-NH` *r*:sub:`lower` *r*:sub:`upper` *p* looptime

.. csv-table::
    :file: en_file_entries.csv
    :widths: 7 28 6 5
    :header-rows: 1

.. note:: 

    The last column of the table denotes when the respective quantity is printed to the ``.en`` file. The letters have the following meaning:

        | **A**:   Always printed
        | **QM**:  Only printed if QM atoms are present
        | **MM**:  Only printed if MM atoms are present
        | **MM***: Only printed if MM atoms are present, which are not treated by the :ref:`guffdatFile`
        | **P**:   Only printed if pressure coupling is enabled *via* the :ref:`pressureCouplingKeys` in the ``.in`` file
        | **NH**:  Only printed if the Nose Hoover chain thermostat is enabled *via* the :ref:`thermostatKey` key in the ``.in`` file
        | **DC**:  Only printed if distance constraints are applied
    
    If the interval at which the results are printed to the output files is changed *via* the :ref:`outputfreqKey` key in the ``.in`` file, 
    the values in the ``.en`` file are averaged over the respective interval.
    
    All printed quantities in correct ordering and with associated units are given in the ``.info`` output file, which 
    is described in section :ref:`infoFile`.

.. _forceFile:

***********
Force File
***********

**File Type:** ``.force``

Stores the force *F* acting on each atom for every frame of the simulation in the following format:
    
    | **line 1:** n_atoms *a* *b* *c* *α* *β* *γ*
    | **line 2:** total_force
    | **line 3 to (n_atoms + 2):** atom_type_name *F*:sub:`x` *F*:sub:`y` *F*:sub:`z`

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters in units of Å and degrees. The second line gives the total force acting on the system in 
:math:`\frac{\text{kcal}}{\text{mol Å}}`. The following lines contain the name of the atom type (as given in the 
:ref:`moldescriptorFile` file) and the associated forces acting along the x, y and z direction in 
:math:`\frac{\text{kcal}}{\text{mol Å}}` for each atom in the system.

.. note::

    The total force should be marginal if no external fields (Planned) are applied to the system.

.. _infoFile:

**********
Info File
**********

**File Type:** ``.info``

Stores information about various quantities of the system and their units for the last frame calculated. The quantities are identical 
to those in the ``.en`` file (described under section :ref:`energyFile`), except the first entry which is the total simulation time 
in ps instead of the step number. Entries in the ``.info`` file are read from left to right and top to bottom.

.. _instantEnergyFile:

********************
Instant Energy File
********************

**File Type:** ``.instant_en``

Stores the exact same quantities as the ``.en`` file (described under section :ref:`energyFile`), but for the last frame calculated
and not averaged over the interval set by the :ref:`outputfreqKey` key in the ``.in`` file. Therefore, it is identical to the ``.en``
file if the output frequency is set to 1.

.. _logFile:

*********
Log File
*********

**File Type:** ``.log``

Starts with general information about the **PQ** program, such as the author, version, and the date of compilation. The file then tracks the 
initialization of **PQ** and the simulation settings used. In case of a successful simulation, the file ends with the text 'PQ ended normally'. 
In case of an error, the file shows the respective error message.

.. _momentumFile:

**************
Momentum File
**************

**File Type:** ``.mom``

Stores the total momentum *p* and the total angular momentum *L* of the system as well as their individual *x*, *y* and *z* components for 
every frame in the following format:

    step_number *p* *p*:sub:`x` *p*:sub:`y` *p*:sub:`z` *L* *L*:sub:`x` *L*:sub:`y` *L*:sub:`z`

The total momentum and its components are given in units of :math:`\frac{\text{amu Å}}{\text{fs}}`, while the total angular momentum and its 
components are given in units of :math:`\frac{\text{amu }\text{Å}^2}{\text{fs}}`.

.. _refFile:

***************
Reference File
***************

**File Type:** ``.ref``

Lists the references to be cited when publishing results obtained *via* the chosen simulation settings as regular text and in BibTeX format.

.. _restartFile:

*************
Restart File
*************

**File Type:** ``.rst``

Stores the coordinates, velocities, and forces of each atom for the last performed simulation step in the following format:
    
    | **line 1:** "Step" step_number
    | **line 2:** "Box" *a* *b* *c* *α* *β* *γ*
    | **line 3 to (n_atoms + 2):** element running_index moltype *x* *y* *z* *v*:sub:`x` *v*:sub:`y` *v*:sub:`z` *F*:sub:`x` *F*:sub:`y` *F*:sub:`z`

The first line contains the string "Step" followed by the total number of performed simulation steps. The second line starts with the string 
"Box" followed by the parameters *a*, *b*, *c*, *α*, *β*, and *γ*, which denote the parameters of the simulation box in units of Å and degrees. 
The following lines contain the element symbol, a running index just for human readability, the moltype the atom belongs to according to the 
:ref:`moldescriptorFile` setup file, the Cartesian coordinates in Å, the velocities in :math:`\frac{\text{Å}}{\text{fs}}`, and the forces in 
:math:`\frac{\text{kcal}}{\text{mol Å}}` for each atom in the system. The moltype value is set to 0 if no :ref:`moldescriptorFile` file is used.

.. note::

    If the thermostat is set to the Nose Hoover chain *via* the :ref:`thermostatKey` key in the ``.in`` file, as many additional lines as 
    chosen number of heat baths (set *via* the :ref:`nhchainlenghtKey` key) are added to the restart file between the box and the atom 
    information in the following format:

        "Chi" level *χ* *ζ*

    Every line associated with the Nose Hoover chain thermostat begins with the string "Chi" followed by the level of the thermostat,
    the corresponding friction coefficient *χ* and the cumulant *ζ*. Since these are time-dependent variables, they are
    printed to the restart file.

.. attention::

    A ``.rst`` file needs to be provided by the user for the first run of the simulation alongside the :ref:`Input File <inputFile>`. 
    Furthermore, this first ``.rst`` file has to contain all atoms of a moltype in the same order as provided in the 
    :ref:`moldescriptorFile` setup file. In this first ``.rst`` file, the whole first line denoting the step number as well as the 
    columns denoting the velocities and forces can be omitted. If the parameters defining the box angles are left out, they are 
    assumed to be 90°. Also the information about the Nose Hoover chain thermostat is not needed for the first ``.rst`` file and 
    can be omitted. The whole line defining the box can be left out as well if the box is cubic and its density is provided *via* the
    :ref:`densityKey` keyword in the ``.in`` file.

.. _stressFile:

************
Stress File
************

**File Type:** ``.stress``

Stores the individual components of the stress tensor *σ* for every frame in the following format:
    
    step_number *σ*:sub:`ax` *σ*:sub:`ay` *σ*:sub:`az` *σ*:sub:`bx` *σ*:sub:`by` *σ*:sub:`bz` *σ*:sub:`cx` *σ*:sub:`cy` *σ*:sub:`cz`

The individual components of the stress tensor are given in units of :math:`\frac{\text{kcal}}{\text{mol Å}^3}`. The first character
of the index denotes the lattice parameter (*a*, *b*, *c*), while the second character denotes the respective lattice parameter component 
(*x*, *y*, *z*).

.. _timingFile:

**************
Timing File
**************

**File Type:** ``.timings``

Tracks the time **PQ** takes for executing the individual parts of the simulation.

.. _trajectoryFile:

***************
Trajectory File
***************

**File Type:** ``.xyz``

Stores the coordinates (*x*, *y*, *z*) of each atom for every frame of the simulation in the following format:
    
    | **line 1:** n_atoms *a* *b* *c* *α* *β* *γ*
    | **line 2:** empty
    | **line 3 to (n_atoms + 2):** atom_type_name *x* *y* *z*

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation
box and the respective box parameters in units of Å and degrees. The second line is left empty. The following lines contain the name 
of the atom type (as given in the :ref:`moldescriptorFile` file) and the associated Cartesian coordinates in Å for each atom in the system.

.. _velocityFile:

*************
Velocity File
*************

**File Type:** ``.vel``

Stores the velocity *v* of each atom for every frame of the simulation in the following format:
    
    | **line 1:** n_atoms *a* *b* *c* *α* *β* *γ*
    | **line 2:** empty
    | **line 3 to (n_atoms + 2):** atom_type_name *v*:sub:`x` *v*:sub:`y` *v*:sub:`z`

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation
box and the respective box parameters in units of Å and degrees. The second line is left empty. The following lines contain the name 
of the atom type (as given in the :ref:`moldescriptorFile` file) and the associated velocities along the x, y and z direction in 
:math:`\frac{\text{Å}}{\text{fs}}` for each atom in the system.

.. _virialFile:

************
Virial File
************

**File Type:** ``.vir``

Stores the individual components of the virial tensor *W* for every frame in the following format:
    
    step_number *W*:sub:`ax` *W*:sub:`ay` *W*:sub:`az` *W*:sub:`bx` *W*:sub:`by` *W*:sub:`bz` *W*:sub:`cx` *W*:sub:`cy` *W*:sub:`cz`

The individual components of the virial tensor are given in units of :math:`\frac{\text{kcal}}{\text{mol}}`. The first character
of the index denotes the lattice parameter (*a*, *b*, *c*), while the second character denotes the respective lattice parameter component 
(*x*, *y*, *z*).