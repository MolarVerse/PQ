.. _outputFiles:

############
Output Files
############

.. _boxFile:

*********
Box File
*********

File Type: ``.box``

Contains the three lattice parameters (*a*, *b*, *c*) and the three angles (*α*, *β*, *γ*) of the simulation box for every frame in the following format:
    
    step_number *a* *b* *c* *α* *β* *γ*

.. _chargeFile:

************
Charge File
************

File Type: ``.chrg``

Stores the charge of each atom for every frame of the simulation in the following format:
    
    | line 1: number_of_atoms *a* *b* *c* *α* *β* *γ*
    | line 2: empty
    | line 3 to (number_of_atoms + 2): element charge

