.. _developerGuide:

###############
Developer Guide
###############

This section includes information for developers who want to contribute to the project. It includes information about the project structure, how to run the tests, and how to build the documentation. It also includes information about the project's coding style and how to contribute to the project.

*****************
Project Structure
*****************

**PQ** is a C++ molecular dynamics simulation engine with Python bindings, organized into a modular architecture. The project follows standard C++ project conventions with clear separation between headers, source code, applications, tests, and documentation.

====================
Core Directory Layout
====================

The main directories are organized as follows:

**Source Code:**
    - ``src/`` - Implementation files (.cpp) containing the core functionality
    - ``include/`` - Header files (.hpp) defining interfaces and declarations
    - ``apps/`` - Main application executable (PQ.cpp)

**Build System:**
    - ``CMakeLists.txt`` - Main CMake configuration file

**Testing:**
    - ``tests/`` - Unit tests for individual components
    - ``integration_tests/`` - Integration tests using pytest

**Documentation:**
    - ``docs/`` - Sphinx documentation source files and build configuration
    - ``README.md`` - Project overview and installation instructions

**External Dependencies:**
    - ``external/`` - Third-party libraries (googletest, progressbar, toml++)

**Utilities:**
    - ``scripts/`` - Build scripts, Singularity definition files, and utility scripts
    - ``examples/`` - Example simulation input files
    - ``config/`` - License headers

=================
Core Architecture
=================

The **PQ** codebase is organized into functional modules, each contained in its own directory within both ``src/`` and ``include/``:

**Simulation Engine:**
    - ``engine/`` - Main simulation engine and orchestration
    - ``integrator/`` - Numerical integration algorithms
    - ``thermostat/`` - Temperature control algorithms
    - ``manostat/`` - Pressure control algorithms

**Physical Models:**
    - ``forceField/`` - MM bonded interactions
    - ``intraNonBonded/`` - MM intramolecular non-bonded interactions
    - ``potential/`` - MM intermolecular non-bonded interactions
    - ``QM/`` - Interface to QM runner programs

**System Setup:**
    - ``input/`` - Input and setup file parsing
    - ``setup/`` - System initialization and configuration
    - ``simulationBox/`` - Simulation cell, molecule and atom handling
    - ``connectivity/`` - Molecular topology and bonding information

**Data Management:**
    - ``box/`` - Simulation box geometry and periodic boundary conditions
    - ``physicalData/`` - Physical constants and unit conversions
    - ``output/`` - Output file generation and data writing

**Computational Infrastructure:**
    - ``linearAlgebra/`` - Vector and matrix operations
    - ``utilities/`` - General utility functions and helpers
    - ``kernels/`` - Computational kernels and optimized routines
    - ``timings/`` - Performance profiling and timing utilities

**Advanced Features:**
    - ``constraints/`` - Constraint algorithms (SHAKE, RATTLE, etc.)
    - ``maxwellBoltzmann/`` - Maxwell-Boltzmann velocity initialization
    - ``resetKinetics/`` - Kinetic energy manipulation
    - ``virial/`` - Virial stress tensor calculations
    - ``mpi/`` - MPI parallelization support
    - ``python/`` - Python bindings and interface
    - ``opt/`` - Optimization algorithms

**Design Patterns:**
    - ``concepts/`` - C++20 concepts for template constraints
    - ``exceptions/`` - Custom exception classes
    - ``settings/`` - Configuration and settings management

===================
Build Configuration
===================

The project uses CMake as its build system:

    - Root ``CMakeLists.txt`` configures the overall project
    - Each major directory contains its own ``CMakeLists.txt`` for modular compilation
    - External dependencies are managed through CMake's FetchContent or find_package
    - Support for different build types (Debug, Release) and optional features (MPI support)

**************
Software Tests
**************

============
Unit Testing
============

Unit tests are located in the ``PQ/tests/`` directory.
After building the project, the unit tests can be run by executing the command ``make test`` from within the build directory.

===================
Integration Testing
===================

Integration tests are located in the ``PQ/integration_tests/`` directory.
To run these tests, ensure the following Python packages are installed:

    - `pytest      <https://pypi.org/project/pytest/>`_
    - `pytest-cov  <https://pypi.org/project/pytest-cov/>`_
    - `pqanalysis  <https://pypi.org/project/pqanalysis/>`_
    - `ase         <https://pypi.org/project/ase/>`_
    - `pyscf       <https://pypi.org/project/pyscf/>`_

In addition, the `DFTB+ <https://dftbplus.org/index.html>`_ program package has to be installed.
You can then run the integration tests with the command ``pytest PQ/integration_tests``.

==============
GitHub Actions
==============

The software workflow --- from building the project to running unit and integration tests --- is validated by continuous integration (CI) using GitHub Actions.
The corresponding workflow configuration files are located in the ``PQ/.github/workflows/`` directory.

*************
Documentation
*************

This documentation is written as `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ files ``.rst`` and converted to HTML website files by `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ .
The respective source files are located in ``PQ/docs/sphinx/src/``.
In order to compile the files locally you need to install the following Python packages:

    - `Sphinx            <https://pypi.org/project/Sphinx/>`_
    - `sphinx_sitemap    <https://pypi.org/project/sphinx-sitemap/>`_
    - `breathe           <https://pypi.org/project/breathe/>`_
    - `sphinx_rtd_theme  <https://pypi.org/project/sphinx-rtd-theme/>`_

The project is built by running ``make html`` in the folder ``PQ/docs/sphinx/``.
The resulting website can be viewed by opening the file ``PQ/docs/sphinx/_build/html/index.html`` *via* your favorite browser.

*****************
How to Contribute
*****************

For anyone willing to contribute to the project, it is important to understand the branching model used by the project. The project uses the `Gitflow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. In order to contribute to the project, please follow these steps:


    #. Fork the project on GitHub. (not necessary if you are a member of the project)

    #. Clone your fork locally:
    
        .. code:: bash

            $ git clone https://github.com/MolarVerse/PQ.git

    #. Initialize git flow with the following settings (if not specified default settings are used)

        .. code:: bash

            [master] main
            [develop] dev
            [version tag prefix] v

    #. Create a feature branch for your contribution:
    
        .. code:: bash

            $ git flow feature start <feature_branch_name>


    #. Commit your changes to your feature branch and publish your feature branch:
    
        .. code:: bash

            $ git add <files>
            $ git commit -m "commit message"
            $ git flow feature publish <feature_branch_name>
    
    #. Create a pull request on GitHub.
