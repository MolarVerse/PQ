.. _userG_installation:

############
Installation
############

********************
Building from Source
********************

Create a build directory and navigate into this directory. Within this directory configure cmake:

.. code:: bash

    cmake ../ -DCMAKE_BUILD_TYPE=Release

Optionally it is also possible to enable MPI for Ring Polymer MD

.. code:: bash

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For compilation then type:

.. code:: bash

    make -j<#procs>

.. _singularity:

***********
Singularity
***********

The :code:`scripts` directory contains a file called :code:`pimd_qmcf.def`, which is a definition file for a singularity container. First build this container as following

.. code:: bash

    singularity build --fakeroot <name of container>.sif pimd_qmcf2.def

In order to run the application which is build within the container just execute

.. code:: bash
    
    singularity run --no-home <name of container>.sif <input file>