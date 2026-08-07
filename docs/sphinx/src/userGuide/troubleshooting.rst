.. _troubleshooting:

###############
Troubleshooting
###############

This page lists common setup failures and the input file area to check first.

Existing Output Files
*********************

PQ does not overwrite output files by default. If setup stops with
``File already exists - filename = ...``, either remove the previous output
files, change :ref:`file_prefix <fileprefixkey>`, or set:

.. code-block:: text

    overwrite_output = true;

Missing Setup Files
*******************

Errors such as ``Cannot open start file``, ``Cannot open topology file`` or
``Cannot open parameter file`` mean that the path in the input file cannot be
opened from the current working directory. Run examples from their own
directory, or use explicit paths for the corresponding setup-file key.

Some setup files are only required for specific modes:

    | ``topology_file`` is required when ``force-field`` is set to ``bonded`` or ``on``, or when ``shake`` is set to ``on`` or ``shake``.
    | ``parameter_file`` is required when ``force-field`` is set to ``bonded`` or ``on``.
    | ``mshake_file`` is required when ``shake = mshake`` is selected.
    | ``dftb_file`` is required for direct DFTB+ calculations.

Unknown Job Type
****************

If setup reports an invalid ``jobtype``, use one of the supported job types:

    | ``mm-md``
    | ``qm-md``
    | ``qm-rpmd``
    | ``mm-opt``

ASE and Slater-Koster Files
***************************

Built-in Slater-Koster sets for ``ase-dftbplus`` require ASE support at build
time. If PQ was built without ASE support and the input requests ``slakos = 3ob``
or ``slakos = matsci``, rebuild with:

.. code-block:: bash

    $ cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_ASE=On

For custom Slater-Koster files, set ``slakos = custom`` and provide
``slakos_path``.

Kokkos Fallbacks
****************

When PQ is built with Kokkos support, the accelerated MM path is only used for
supported non-Coulombic and Coulomb settings. If another combination is
requested, PQ prints a warning and falls back to serial execution.

Constraint Setup
****************

For ``shake = shake`` or ``shake = mshake``, check that:

    | ``topology_file`` is set when SHAKE bond constraints are needed.
    | ``mshake_file`` is set for M-SHAKE.
    | atom names in the M-SHAKE reference match the referenced molecule type.
    | atom indices in topology files refer to the current structure.
