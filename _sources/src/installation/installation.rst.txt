.. _userG_installation:

############
Installation
############

********************
Building from Source
********************

Create a build directory and navigate into this directory. Within this directory configure cmake:

.. tip::

    cmake ../ -DCMAKE_BUILD_TYPE=Release

Optionally it is also possible to enable MPI for Ring Polymer MD

.. tip::

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For compilation then type:

.. tip::

    make -j<#procs>

.. _singularity:

***********
Singularity
***********

Singularity is a containerization tool that allows to run applications in a container. This is especially useful for running applications on HPC systems where the user does not have root access. Singularity is available on most HPC systems. The PQ software package provides three (hopefully) useful singularity definition files, which can be found in the `<https://github.com/MolarVerse/PQ>`_. 

The file :code:`PQ.def` represents a definition file to build a singularity container based on a fully sequential build of the latest release of PQ. The file :code:`PQ_openmpi.def` is an extension of the previously mentioned definition file - including the OpenMPI library of choice and therefore compiled with MPI support. As MPI application are highly restricted regarding the applied MPI version, before building the container the __VERSION__ in the definition file has to be substituted with the desired OpenMPI version. Therefore, a small and simple bash script :code:`inferOpenMpiVersion.sh` is provided, which automatically substitutes the __VERSION__ with the desired OpenMPI version when given as command line argument or if no CLI argument is given it tries to infer the needed OpenMPI variable from the environment variable :code:`$PATH`.

In order to build both containers from the singularity file the following command can be used:

.. tip::

    singularity build --fakeroot <name_of_container>.sif <name_of_definiton_file>.def

In order to execute the program *via* the singularity container two possible commands are shown below:

.. tip::

    singularity exec --env MYPATH=$PATH <name_of_container>.sif /data/PQ/build/apps/PQ <name_of_input_file>

.. tip::
    
     singularity run --env MYPATH=$PATH <name_of_container>.sif <name_of_input_file>

Depending on the directory structure of the host system it might be necessary to bind/mound the directory containing the input file to the container. This can be achieved by adding :code:`--bind $PWD` to the singularity command.

The third definiton file is only experimental at the moment as it is used in combination with a miniconda environment in the container. This should make it possible in future releases to build the singularity container based on an environment.yml file. This definiton file is called :code:`PQ_conda.def`.