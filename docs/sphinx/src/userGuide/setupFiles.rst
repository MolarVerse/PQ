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
the ``.rst`` file as described in the :ref:`restartFile` section.  The moldescriptor file is structured into groups for every moltype,
which have the following form:

    | **line 1:** name n_atoms charge
    | **line 2 to (n_atoms + 1):** atom_type_name atom_type_index point_charge global_vdW_index

The parameters name, n_atoms, and charge in the first line denote the name of the moltype, the number of atoms in the moltype, and the total
charge of the moltype in units of the elementary charge *e*. The following lines contain the name of the atom type, the index of the MM 
atom type, the MM point charge in units of *e* and the global van der Waals index for each atom in the moltype. The atom type name is 
irrelevant for internal calculations, but will be printed to various :ref:`outputFiles`. The index of the atom type is relevant for 
defining intra- and intermolecular non-bonded interactions in the :ref:`guffdatFile`. The point charge as well as the global van der Waals 
type are only relevant for MM atoms that are not treated *via* the :ref:`guffdatFile`. The global van der Waals index is used to assign 
identical atoms from different moltypes to the same atom type.

.. Attention::

    Providing a moldescriptor file is optional for pure QM calculations, but becomes mandatory if there are MM atoms present in the 
    system and/or pressure coupling is enabled *via* the :ref:`pressureCouplingKeys` in the ``.in`` file. If no moldescriptor file is
    provided, the element symbol (as given by the :ref:`restartFile`) will be printed to the output files instead of the 
    atom type name. In case of a pure QM calculation the charge of the moltype, the atom_type_index as well as the point_charge can be 
    set to 0 and the global_vdW_type can be omitted. Moreover, the  For MM calculations that utilize just the :ref:`guffdatFile`, the 
    point_charge can be set to 0 and the global_vdW_type can be omitted.

.. _guffdatFile:

**********
GUFF File
**********

**Default Name:** ``guff.dat``

The grand unified force field (GUFF) file can be used to define the non-bonding force field parameters for the MM atoms in the system. 
The GUFF file defines the Coulomb potential *V*:sub:`Coulomb` *via* equation :eq:`guffCoulombEquation` and the non-Coulombic potential *V*:sub:`non-Coulomb` 
*via* the generalized equation :eq:`guffNonCoulombEquation` for every atom type index in every moltype of the system in the following format:

    | moltype_1; atom_type_index_1; moltype_2; atom_type_index_2; *r*:sub:`cut`; *c*:sub:`0`; *c*:sub:`1`; *c*:sub:`2`; *c*:sub:`3`; *c*:sub:`4`; *c*:sub:`5`; *c*:sub:`6`; *c*:sub:`7`; *c*:sub:`8`; *c*:sub:`9`; *c*:sub:`10`; *c*:sub:`11`; *c*:sub:`12`; *c*:sub:`13`; *c*:sub:`14`; *c*:sub:`15`; *c*:sub:`16`; *c*:sub:`17`; *c*:sub:`18`; *c*:sub:`19`; *c*:sub:`20`; *c*:sub:`21`; *c*:sub:`22`

    .. math:: V_{\text{Coulomb}} = \frac{c_0}{r}
        :label: guffCoulombEquation

    .. math:: V_{\text{non-Coulomb}} = \frac{c_1}{r^{c_2}} + \frac{c_3}{r^{c_4}} + \frac{c_5}{r^{c_6}} + \frac{c_7}{r^{c_8}} + \frac{c_9}{1 + e^{c_{10} (r - c_{11})}} + \frac{c_{12}}{1 + e^{c_{13} (r - c_{14})}} + c_{15} e^{c_{16} (r - c_{17})^{c_{18}}} + c_{19} e^{c_{20} (r - c_{21})^{c_{22}}}
        :label: guffNonCoulombEquation
        
The moltype and atom_type_index are defined as in the :ref:`moldescriptorFile` file.
Utilizing equation :eq:`guffNonCoulombEquation`, Lennard-Jones, Buckingham and Morse potentials, as well as arbitrary combinations of them can be used 
for the description of the non-Coulombic interactions. The parameter *r*:sub:`cut` gives the cutoff radius for the non-bonded interactions. Distances 
are given in Å and energies in kcal/mol. The units of the parameters are chosen accordingly.


.. Attention::

    All entries in the GUFF file need to be separated *via* a semicolon ``;``. Furthermore, defining all possible interactions is mandatory. If a certain 
    potential is not needed, the corresponding coefficients are set to 0.

    Using the GUFF file requires the :ref:`moldescriptorFile` setup file to be provided as well.
.. _topologyFile:

*************
Topology File
*************

.. _parameterFile:

**************
Parameter File
**************

.. _mshakeFile:

************
M-SHAKE File
************