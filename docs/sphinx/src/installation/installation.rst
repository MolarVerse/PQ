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

Coming soon...