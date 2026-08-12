.. _userG_installation:

############
Installation
############

********************
Building from Source
********************

**Prerequisites**

* CMake >= 3.18
* GCC >= 13.0

Clone the PQ GitHub repository and navigate into the directory:

.. code-block:: bash

    $ git clone https://github.com/MolarVerse/PQ.git
    $ cd PQ

Create a build directory and navigate into this directory:

.. code-block:: bash

    $ mkdir build
    $ cd build

Within this directory configure CMake:

.. code-block:: bash

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release

Then compile the project:

.. code-block:: bash

    $ make -j <number_of_processors>

The executable is written to ``build/apps/PQ``.

Build Options
=============

Common CMake options are listed below. Boolean options are set with ``On`` or
``Off``.

.. list-table::
    :header-rows: 1
    :widths: 28 14 58

    * - Option
      - Default
      - Purpose
    * - ``CMAKE_BUILD_TYPE``
      - ``RelWithDebug``
      - Build type. Supported values are ``Debug``, ``RelWithDebug`` and ``Release``.
    * - ``BUILD_WITH_TESTS``
      - ``On``
      - Build the C++ unit tests.
    * - ``BUILD_WITH_MPI``
      - ``Off``
      - Enable MPI support, mainly used for ring-polymer QM-MD.
    * - ``BUILD_WITH_ASE``
      - ``On``
      - Build ASE-based QM runners and built-in Slater-Koster setup.
    * - ``PQ_SLAKOS_SOURCE_DIR``
      - unset
      - Use preseeded ``3ob`` and ``matsci`` directories instead of cloning them.
    * - ``BUILD_WITH_PYTHON_BINDINGS``
      - ``Off``
      - Build Python bindings.
    * - ``BUILD_WITH_NATIVE``
      - ``On``
      - Optimize release builds for the local CPU. Disable for portable binaries.
    * - ``BUILD_WITH_LTO``
      - ``Off``
      - Enable link-time optimization for release builds.

Example MPI build:

.. code-block:: bash

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For a network-restricted ASE build, prepare a directory containing ``3ob`` and
``matsci`` checkouts, then configure with:

.. code-block:: bash

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release \
        -DPQ_SLAKOS_SOURCE_DIR=/path/to/slakos

.. _singularity:

***********
Singularity
***********

Singularity is a containerization tool that allows to run applications in a container. This is especially useful 
for running applications on HPC systems where the user does not have root access. Singularity is available on most 
HPC systems. The PQ software package provides three Singularity definition
files in ``PQ/scripts/``.

The file ``PQ.def`` represents a definition file to build a singularity container based on a fully sequential build
of the latest release of PQ. The file ``PQ_openmpi.def`` is an extension of the previously mentioned definition 
file - including the OpenMPI library of choice and therefore compiled with MPI support. As MPI applications are highly
restricted regarding the applied MPI version, before building the container the __VERSION__ in the definition file 
has to be substituted with the desired OpenMPI version. Therefore, a small and simple bash script ``inferOpenMpiVersion.sh`` 
is provided, which automatically substitutes the __VERSION__ with the desired OpenMPI version when given as command 
line argument or if no CLI argument is given it tries to infer the needed OpenMPI variable from the environment variable ``$PATH``.

In order to build both containers from the singularity file the following command can be used:

.. code-block:: bash

    $ singularity build --fakeroot <name_of_container>.sif <name_of_definition_file>.def

In order to execute the program *via* the singularity container two possible commands are shown below:

.. code-block:: bash

    $ singularity exec --env MYPATH=$PATH <name_of_container>.sif /data/PQ/build/apps/PQ <name_of_input_file>

.. code-block:: bash
    
     $ singularity run --env MYPATH=$PATH <name_of_container>.sif <name_of_input_file>

Depending on the directory structure of the host system it might be necessary to bind/mount the directory containing the
input file to the container. This can be achieved by adding ``--bind $PWD`` to the singularity command.

The third definition file is experimental at the moment as it is used in combination with a miniconda environment in the
container. This should make it possible in future releases to build the singularity container based on an environment.yml file. 
This definition file is called ``PQ_conda.def``.
