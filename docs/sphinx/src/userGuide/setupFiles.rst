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
identical elements exposed to a similar chemical environment from different moltypes to the same atom type.

.. Attention::

    Providing a moldescriptor file is optional for pure QM calculations, but becomes mandatory if there are MM atoms present in the 
    system and/or pressure coupling is enabled *via* the :ref:`pressureCouplingKeys` in the ``.in`` file. If no moldescriptor file is
    provided, the element symbol (as given by the :ref:`restartFile`) will be printed to the output files instead of the 
    atom type name. In case of a pure QM calculation the charge of the moltype, the atom_type_index as well as the point_charge can be 
    set to 0 and the global_vdW_type can be omitted. For MM calculations that utilize just the :ref:`guffdatFile`, the 
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


.. _dftbFile:

***************
DFTB Setup File
***************

**Default Name:** ``dftb_in.template``

The DFTB setup file is used by PQ to generate the dftb_in.hsd file, which is used as input for calculations by the `DFTB+ <https://dftbplus.org/index.html>`__ software.
As such, it has the same structure and keywords as the human-readable structured data (HSD) file for a single point calculation.
The documentation of which can be found `here <https://www.dftbplus.org/documentation.html>`__.
There is an additional keyword, named ``__GUESS__``, that can be used within the Hamiltonian of the DFTB setup file.
If the ``__GUESS__`` flag is included, the charges will be read for an initial guess in every step of the MD simulation except the first.

.. Attention::

    Providing a DFTB setup file is mandatory if the :ref:`qm_prog <qmprogamKey>` keyword is set to ``dftbplus``.


.. _topologyFile:

*************
Topology File
*************

The topology file defines bonded force-field terms and distance constraints by
global atom index. It is a section-based file. A section starts with a section
keyword and ends with ``END``. Lines starting with ``#`` are ignored.

The accepted section keywords are ``SHAKE``, ``BONDS``, ``ANGLES``,
``DIHEDRALS``, ``IMPROPERS``, ``DIST_CONSTRAINTS`` and ``J_COUPLINGS``. Section
keywords are case-insensitive, and dashes in the keyword are treated like
underscores.

The topology line formats are:

    | ``SHAKE``: atom_1 atom_2 distance [unused_fourth_field]
    | ``BONDS``: atom_1 atom_2 bond_type [``*``]
    | ``ANGLES``: atom_1 atom_2 atom_3 angle_type [``*``]
    | ``DIHEDRALS``: atom_1 atom_2 atom_3 atom_4 dihedral_type [``*``]
    | ``IMPROPERS``: atom_1 atom_2 atom_3 atom_4 improper_type
    | ``DIST_CONSTRAINTS``: atom_1 atom_2 lower_distance upper_distance force_constant dforce_constant_dt
    | ``J_COUPLINGS``: atom_1 atom_2 atom_3 atom_4 j_coupling_type

The optional fourth field in ``SHAKE`` entries is accepted for legacy input
files but is not used by the reader. The optional ``*`` marker in the bonded
sections marks a linker interaction. Topology atom indices are global atom
indices from the current structure.

.. code-block:: text

    SHAKE
    1 2 1.0
    END

    BONDS
    1 2 1
    END

.. _parameterFile:

**************
Parameter File
**************

The parameter file defines the parameters used by the force-field sections in
the :ref:`topologyFile`. Like the topology file, it is a section-based file and
every section is terminated with ``END``. Lines starting with ``#`` are ignored.

The accepted section keywords are ``TYPES``, ``BONDS``, ``ANGLES``,
``DIHEDRALS``, ``IMPROPERS``, ``J_COUPLINGS`` and ``NONCOULOMBICS``. Section
keywords are case-insensitive, and dashes in the keyword are treated like
underscores.

The parameter line formats are:

    | ``TYPES``: dummy dummy dummy dummy dummy dummy scale_14_coulomb scale_14_vdw
    | ``BONDS``: bond_type equilibrium_distance force_constant
    | ``ANGLES``: angle_type equilibrium_angle force_constant
    | ``DIHEDRALS``: dihedral_type force_constant periodicity phase
    | ``IMPROPERS``: improper_type force_constant periodicity phase
    | ``J_COUPLINGS``: j_coupling_type j_0 force_constant a b c phase [symmetry]

The ``NONCOULOMBICS`` header may be followed by ``LJ``, ``BUCKINGHAM`` or
``MORSE``. If no type is given, ``LJ`` is used.

    | ``NONCOULOMBICS LJ``: atom_type_1 atom_type_2 c6 c12 [cutoff]
    | ``NONCOULOMBICS BUCKINGHAM``: atom_type_1 atom_type_2 a d_rho c6 [cutoff]
    | ``NONCOULOMBICS MORSE``: atom_type_1 atom_type_2 dissociation_energy well_width equilibrium_distance [cutoff]

If the optional non-Coulombic cutoff is omitted or set to a negative value, the
global Coulomb cutoff is used. Angle and phase values in the parameter file are
read in degrees.

.. code-block:: text

    BONDS
    1 1.2 1.3
    END

    NONCOULOMBICS LJ
    1 2 0.1 1.2 12.0
    END

.. _mshakeFile:

************
M-SHAKE File
************

The M-SHAKE file defines one reference geometry per molecule type for the
``mshake`` constraint mode. It uses repeated extended-XYZ-like blocks:

.. code-block:: text

    3
    moltype = 1;
    H 0.0 0.0 0.0
    O 1.0 1.0 1.0
    H 2.0 2.0 2.0

The first line of each block is the number of atoms in the reference geometry.
The second line must contain ``moltype = <id>;`` and is parsed with the same
command syntax as the input file. It is followed by one atom line per reference
atom: atom name and Cartesian coordinates. The atom names must match the atom
names of the referenced molecule type.
