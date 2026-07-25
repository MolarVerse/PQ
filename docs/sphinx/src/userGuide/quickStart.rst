.. _quickStart:

###########
Quick Start
###########

This page shows the shortest path from a source checkout to a local **PQ** run.
For full build details see :ref:`userG_installation`; for input syntax see
:ref:`inputFile`.

Build PQ
********

From the repository root, configure and build a release binary:

.. code-block:: bash

    $ mkdir -p build
    $ cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release
    $ make -j <number_of_processors>

The executable is written to ``build/apps/PQ``.

Run an Example
**************

The ``examples/h2o_mm`` example is a self-contained molecular mechanics run. It
uses only files stored in the example directory.

.. code-block:: bash

    $ cd ../examples/h2o_mm
    $ ../../build/apps/PQ run-01.in

The input file sets ``file_prefix = h2o-md-01``. Output files therefore start
with ``h2o-md-01`` and use the extensions described in :ref:`outputFiles`.

For a short smoke test, copy the example directory first and reduce ``nstep`` in
``run-01.in`` before running it.

Inspect the Input
*****************

A PQ run is controlled by one input file. The water example uses:

    | ``jobtype = mm-md`` for a molecular-mechanics MD simulation
    | ``start_file = input_h2o.rst`` for the initial structure
    | default ``moldescriptor_file`` and ``guff_file`` setup files
    | ``file_prefix = h2o-md-01`` for generated output names

Additional setup files and output formats are documented in the
:ref:`referenceManual`.

Next Steps
**********

Use :ref:`examples` to choose a closer starting point for QM, MACE, ASE-DFTB+,
ASE-xTB or RPMD calculations. If the run stops during setup, check
:ref:`troubleshooting` first; most setup failures are missing files, conflicting
output files or optional build dependencies.
