.. _generalUsage:

#############
General Usage
#############

**PQ** is based on a single input file execution. A fully prepared setup is completely defined by an input 
file ``PQ.in`` and can be executed in the following form:

.. code-block:: bash

    $ PQ PQ.in

Run **PQ** from the directory that contains the input file and any setup files
referenced with relative paths. The input file selects the job type, physical
model, output names and required setup files.

For a first complete run, see :ref:`quickStart`. For the full input syntax and
file formats, see :ref:`referenceManual`.

Inspect and Validate
********************

Use the command-line metadata to check the installed PQ build:

.. code-block:: bash

    $ PQ --version
    $ PQ --capabilities=json

Before starting a simulation, validate the input from its working directory:

.. code-block:: bash

    $ PQ --validate PQ.in
    $ PQ --validate PQ.in --format=json
    $ PQ --validate PQ.in --scope=portable --format=json

The default ``installed`` scope checks input syntax, setting dependencies,
compiled capabilities and required files. The ``portable`` scope checks syntax
and setting dependencies without requiring the target build or local files.
This is useful for input generators and setup packages. Validation does not
start a simulation or create output. Referenced file contents are checked when
the simulation starts.

Exit status ``0`` means the input is valid, including inputs with warnings.
Status ``1`` means the input is invalid. Operational validation failures use
status ``2``.
