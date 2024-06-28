############
Output Files
############

*********
Box File
*********

**File Type:** ``.box``

Contains the three lattice parameters (*a*, *b*, *c*) and the three angles (*α*, *β*, *γ*) of the simulation box for 
every frame in the following format:
    
    step_number *a* *b* *c* *α* *β* *γ*

The lattice parameters are given in units of Å and the angles are given in units of degrees.

************
Charge File
************

**File Type:** ``.chrg``

Stores the charge of each atom for every frame of the simulation in the following format:
    
    | line 1: n_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: empty
    | line 3 to (n_atoms + 2): element charge

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters in units of Å and degrees. The second line is left empty. The following lines contain the element 
symbol and its charge in units of the elementary charge *e* for each atom in the system.

***********
Energy File
***********

**File Type:** ``.en``

Stores information about the energy and various other quantities of the system for every frame in the following format:

    step_number *T* *P* *E*:sub:`tot` *E*:sub:`QM` *N*:sub:`QM-atoms` *E*:sub:`kin` *E*:sub:`intra` *V* *ρ* *p* looptime

The parameter *T* denotes the temperature of the system in Kelvin, *P* denotes the pressure in bar, *E*:sub:`tot` denotes the total
energy of the system in :math:`\frac{\text{kcal}}{\text{mol}}`, *E*:sub:`QM` denotes the quantum mechanical energy of the system in
:math:`\frac{\text{kcal}}{\text{mol}}`, *N*:sub:`QM-atoms` denotes the number of atoms treated quantum mechanically, *E*:sub:`kin`
denotes the kinetic energy of the system in :math:`\frac{\text{kcal}}{\text{mol}}`, *E*:sub:`intra` denotes the intramolecular energy
of the system in :math:`\frac{\text{kcal}}{\text{mol}}` (0 for pure QM MD simulations), *V* denotes the volume of the system in Å³, *ρ* 
denotes the density of the system in :math:`\frac{\text{g}}{\text{cm}^3}`, *p* denotes the momentum in :math:`\frac{\text{amu Å}}{\text{fs}}`,
and looptime denotes the time taken to complete the full MD simulation step in s.

.. note:: 

    In case of an *NVE* or *NVT* simulation, the columns for *V* and *ρ* are omitted from the ``.en`` file as they remain constant throughout. 
    All quantities in correct ordering and with associated units are given in the ``.info`` output file, which is described in section `Info File`_.

***********
Force File
***********

**File Type:** ``.force``

Stores the force *F* acting on each atom for every frame of the simulation in the following format:
    
    | line 1: n_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: total_force
    | line 3 to (n_atoms + 2): element *F*:sub:`x` *F*:sub:`y` *F*:sub:`z`

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters in units of Å and degrees. The second line gives the total force acting on the system in 
:math:`\frac{\text{kcal}}{\text{mol Å}}`. The following lines contain the element symbol and the associated forces acting along the 
x, y and z direction in :math:`\frac{\text{kcal}}{\text{mol Å}}` for each atom in the system.

**********
Info File
**********

**File Type:** ``.info``

Stores information about various quantities of the system and their units for the last frame calculated. The quantities are identical to those 
in the ``.en`` file (described under section `Energy File`_), except the first entry which is the total simulation time in ps instead of the step number. 

*********
Out File
*********

**File Type:** ``.out``

Starts with general information about the **PQ** program, such as the author, version, and the date of compilation. The file then tracks the 
initialization of **PQ** and the simulation settings used. In case of a successful simulation, the file ends with the text 'PQ ended normally'. 
In case of an error, the file shows the respective error message.

***************
Reference File
***************

**File Type:** ``.out.ref``

Lists the references to be cited when publishing results obtained *via* the chosen simulation settings.

*************
Velocity File
*************

**File Type:** ``.vel``

Stores the velocity *v* of each atom for every frame of the simulation in the following format:
    
    | line 1: n_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: empty
    | line 3 to (n_atoms + 2): element *v*:sub:`x` *v*:sub:`y` *v*:sub:`z`

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation
box and the respective box parameters in units of Å and degrees. The second line is left empty. The following lines contain the element
symbol and the associated velocities along the x, y and z direction in :math:`\frac{\text{Å}}{\text{fs}}` for each atom in the system.





