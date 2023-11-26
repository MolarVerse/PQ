.. _featureList:

.. role:: red

############
Feature List
############

This is a hopefully up to date list of the features implemented in the current release version of the code.

If the features are marked in red, they are not yet implemented but are planned for the future. There exist
three main categories of not yet implemented features:

1) Features that are planned but not yet implemented (Planned)
2) Features that have a high priority but are not yet implemented (Coming Soon)
3) Features that are in the pipeline but are not yet implemented (In development)


*******
Runners
*******

1) Molecular Mechanics (MM)

    - Classical Molecular Dynamics (MD)
    - :red:`Ring Polymer Molecular Dynamcis (RPMD) - Coming Soon`
    - :red:`Geometry Optimisation (OPT) - In development`
    - :red:`Guffcheck - Coming soon`
    

2) Quantum Mechanics (QM)

    - Classical Molecular Dynamics (MD)
    - Ring Polymer Molecular Dynamcis (RPMD)

3) :red:`Hybrid QM/MM - Planned`

    - Classical Molecular Dynamics (MD)
    - Ring Polymer Molecular Dynamcis (RPMD)

************************
Molecular Mechanics (MM)
************************

At the moment the program supports two different types of force fields
with some further specializations or extensions:

    a) Guff (General Unified Force Field)

        - full Guff equation
        - Lennard Jones quick routine
        - Buckingham quick routine
        - Morse quick routine
        
    b) AMBER type force fields

        - standard AMBER force field with (Lennard Jones non-bonded interactions)
        - Buckingham non-bonded interactions
        - Morse non-bonded interactions

Evaluation Scheme
=================

The evaluation of the forces can be performed *via* the following schemes

1) Brute Force Evaluation
2) Cell List Evaluation
3) :red:`Verlet List with Cell List Evaluation - Coming Soon`

Long Range Corrections
======================

Following long range corrections are implemented:

1) :red:`Ewald Summation - Planned`
2) Wolf Summation
3) :red:`Reaction Field - Coming Soon`

4) :red:`Range separation for non Coulombic interactions - Planned`

Special Moltypes
================

1) :red:`Water - Coming Soon`

*****************
Quantum Mechanics
*****************

At the moment the evaluation of quantum mechanical forces is implemented
for the following QM-engines:

1) DFTB+
2) Turbomole
3) PySCF

******************
Molecular Dynamics
******************

Integrators
===========

1) Velocity Verlet

Thermostats
===========

1) Langevin Thermostat
2) Berendsen Thermostat
3) Velocity Rescaling Thermostat
4) Nose-Hoover Thermostat

Manostats
=========

1) Berendsen Manostat
2) Stochastic Rescaling Manostat

Isotropicity
============

All calculation schemes of any MD runner can be performed with triclinic cells

1) Isotropic
2) Semi-Isotropic
3) Anisotropic (only cell lengths)
4) Full Anisotropic (cell lengths and angles)

Constraints
===========

1) Shake/Rattle
2) :red:`Lincs - Planned`
3) :red:`M-Shake - Planned`

***
MPI
***

At the moment only Ring Polymer Molecular Dynamics (RPMD) is implemented in parallel.
Meaning that each ring polymer can be calculated on a different rank, but the calculation of 
the forces is still performed on a single rank. The only exceptions are the QM-engines, which
are called as external programs and can be run in parallel.


