.. _setupFiles: 

###########
Setup Files
###########

The following setup files can be given as additional input to **PQ**. The names of the used files need to be provided with the according 
:ref:`setupfilekeys` in the ``.in`` file if the name does not match the default name.

.. _moldescriptorFile:

**************
Moldescriptor
**************

**Default Name:** ``moldescriptor.dat``

The moldescriptor file is used to assign every atom in the system to a molecular unit, also called moltype. These molecular units can be as small 
as just a single atom or as big as a whole molecule. They are numbered consecutively starting from 1 and are given in the third column of 
the ``.rst`` file as described in the :ref:`restartFile` section. Providing a moldescriptor file is optional, but becomes mandatory if pressure 
coupling is enabled *via* the :ref:`pressureCouplingKeys` in the ``.in`` file. The moldescriptor file is structured into groups for every moltype,
which have the following form:

    | line 1: name n_atoms charge
    | line 2 to (n_atoms + 1): element atom_type point_charge

The parameters name, n_atoms, and charge in the first line denote the name of the moltype, the number of atoms in the moltype, and the total
charge of the moltype in units of the elementary charge *e*. The following lines contain the element symbol, the MM atom type, and the MM 
point charge in units of *e* for each atom in the moltype. The atom_type as well as the point_charge can be set to 0 in case of a pure
QM calculation.

.. _guffdatFile:

****************
The Guffdat File
****************

**Default Name:** ``guff.dat``



.. _topologyFile:

*****************
The Topology File
*****************

.. _parameterFile:

******************
The Parameter File
******************

.. _mshakeFile:

****************
The M-SHAKE File
****************