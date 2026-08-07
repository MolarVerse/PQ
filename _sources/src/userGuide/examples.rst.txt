.. _examples:

########
Examples
########

The ``examples`` directory contains complete input folders. Each example should
be run from its own directory so relative setup-file paths resolve as intended.

.. list-table::
    :header-rows: 1
    :widths: 22 26 52

    * - Folder
      - Main use case
      - Notes
    * - ``examples/h2o_mm``
      - MM-MD water
      - Self-contained classical MD example using ``moldescriptor.dat`` and ``guff.dat``.
    * - ``examples/amyloid-classical``
      - Classical force-field setup
      - Uses topology, parameter and intramolecular non-bonded setup files.
    * - ``examples/umcm-9_dftb+``
      - DFTB+ with SHAKE
      - Uses ``dftb_in.template`` and a topology file with SHAKE constraints.
    * - ``examples/malondialdehyde_dftb+``
      - DFTB+ QM-MD
      - Uses an external DFTB+ setup template.
    * - ``examples/malondialdehyde_dftb+-rpmd``
      - DFTB+ QM-RPMD
      - Ring-polymer MD example with DFTB+.
    * - ``examples/malondialdehyde_pyscf``
      - PySCF QM-MD
      - Uses the PySCF QM runner.
    * - ``examples/malondialdehyde_tm-rpmd``
      - Turbomole QM-RPMD
      - Uses a Turbomole define template.
    * - ``examples/h2o_mace``
      - MACE water
      - Small MACE example using ``qm_prog = mace_mp``.
    * - ``examples/acof1_mace``
      - MACE solid-state MD
      - MACE example for a covalent organic framework with pressure coupling.
    * - ``examples/mof-5_mace``
      - Custom MACE model
      - Uses ``mace_model_path`` for a custom model URL.
    * - ``examples/mof-5_ase-dftb``
      - ASE-DFTB+
      - Uses the ASE DFTB+ runner and built-in Slater-Koster setup.
    * - ``examples/mof-5_ase-xtb``
      - ASE-xTB
      - Uses the ASE xTB runner with ``xtb-method``.

Running an Example
******************

After building PQ, run an example from the example directory:

.. code-block:: bash

    $ cd examples/h2o_mm
    $ ../../build/apps/PQ run-01.in

Most examples are configured for real simulations rather than minimal smoke
tests. To make a quick local test, copy the example directory and lower
``nstep`` in the copied ``run-01.in``.

Optional Dependencies
*********************

Examples using ``dftbplus``, ``pyscf``, ``turbomole``, ``mace_mp``,
``ase-dftbplus`` or ``ase-xtb`` require the corresponding external program or
Python package to be available in the run environment. ASE-based examples also
require PQ to be built with ASE support.
