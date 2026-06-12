.. _featureList:

.. role:: red

############
Feature List
############

This page summarizes the main features supported by **PQ**. Planned items are
listed separately so they are not confused with currently available run modes.

*********
Run Modes
*********

The supported ``jobtype`` values are:

    | ``mm-md`` - molecular-mechanics molecular dynamics
    | ``qm-md`` - quantum-mechanics molecular dynamics
    | ``qm-rpmd`` - quantum-mechanics ring-polymer molecular dynamics
    | ``mm-opt`` - molecular-mechanics geometry optimization

*******************
Molecular Mechanics
*******************

Force-field models:

    | GUFF (Grand Unified Force Field)
    | AMBER-type force fields
    | Lennard-Jones, Buckingham and Morse non-Coulombic interactions

Force evaluation:

    | brute-force pair evaluation
    | cell-list pair evaluation
    | optional Kokkos acceleration for supported MM Lennard-Jones/Wolf setups

Long-range corrections:

    | no correction
    | Wolf summation

*****************
Quantum Mechanics
*****************

Supported QM runners:

    | DFTB+
    | Turbomole
    | PySCF
    | MACE-MP and MACE-OFF
    | ASE-DFTB+
    | ASE-xTB

******************
Molecular Dynamics
******************

Integrator:

    | Velocity Verlet

Thermostats:

    | Langevin
    | Berendsen
    | stochastic velocity rescaling
    | Nose-Hoover chain

Manostats:

    | Berendsen
    | stochastic cell rescaling

Cell coupling modes:

    | isotropic
    | semi-isotropic
    | anisotropic cell lengths
    | full anisotropic cell lengths and angles

Constraints:

    | SHAKE/RATTLE
    | M-SHAKE
    | distance constraints

************
Optimization
************

Geometry optimization is available for molecular-mechanics calculations. The
supported optimizers are:

    | steepest descent
    | ADAM

***
MPI
***

MPI support is used for QM-RPMD, where individual ring-polymer beads can be
distributed across ranks. Force evaluation itself remains local to each rank,
except for external QM programs that provide their own parallel execution.

*************
Planned Items
*************

The following items are not documented as supported features in this release:

    | hybrid QM/MM job type
    | MM-RPMD
    | Verlet-list force evaluation
    | Ewald summation
    | reaction-field correction
    | LINCS constraints
