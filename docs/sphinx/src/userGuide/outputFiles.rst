.. _outputFiles:

############
Output Files
############

.. _boxFile:

*********
Box File
*********

**File Type:** ``.box``

Contains the three lattice parameters (*a*, *b*, *c*) and the three angles (*α*, *β*, *γ*) of the simulation box for every frame in the following format:
    
    step_number *a* *b* *c* *α* *β* *γ*

.. _chargeFile:

************
Charge File
************

**File Type:** ``.chrg``

Stores the charge of each atom for every frame of the simulation in the following format:
    
    | line 1: n_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: empty
    | line 3 to (n_atoms + 2): element charge

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters. The second line is left empty. The following lines contain the element and charge of each atom in the system.

.. _forceFile:

***********
Force File
***********

**File Type:** ``.force``

Stores the force *F* acting on each atom for every frame of the simulation in the following format:
    
    | line 1: n_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: total_force
    | line 3 to (n_atoms + 2): element *F*:sub:`x` *F*:sub:`y` *F*:sub:`z`

The parameters n_atoms, *a*, *b*, *c*, *α*, *β*, and *γ* in the first line of every frame denote the number of atoms in the simulation 
box and the respective box parameters. The second line gives the total force acting on the system in :math:`\frac{\text{kcal}}{\text{mol Å}}`. 
The following lines contain the element and the associated forces acting along the x, y and z direction of each atom in the system.