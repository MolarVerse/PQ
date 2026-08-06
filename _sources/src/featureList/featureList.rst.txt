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

The supported :ref:`jobtype <jobtype>` values are:

* ``mm-md`` - molecular-mechanics molecular dynamics
* ``qm-md`` - quantum-mechanics molecular dynamics
* ``qm-rpmd`` - quantum-mechanics ring-polymer molecular dynamics
* ``mm-opt`` - molecular-mechanics geometry optimization

*******************
Molecular Mechanics
*******************

**Force-field models**

* :ref:`GUFF <guffdatFile>` (Grand Unified Force Field)
* AMBER-type force fields with :ref:`force-field <forcefieldKey>`
* Lennard-Jones, Buckingham and Morse :ref:`non-Coulombic interactions <noncoulombKey>`

**Force evaluation**

* brute-force pair evaluation
* :ref:`cell-list <celllistKeys>` pair evaluation
* optional Kokkos acceleration for supported MM Lennard-Jones/Wolf setups

**Long-range corrections**

* no correction
* :ref:`Reaction field <longrangeKey>`
* :ref:`Wolf summation <longrangeKey>`

*****************
Quantum Mechanics
*****************

**Supported QM runners**

* :ref:`DFTB+ <qmprogamKey>`
* :ref:`Turbomole <qmprogamKey>`
* :ref:`PySCF <qmprogamKey>`
* :ref:`FeNNol <qmprogamKey>`
* :ref:`MACE-MP and MACE-OFF <qmprogamKey>`
* :ref:`ASE-DFTB+ <qmprogamKey>`
* :ref:`ASE-xTB <qmprogamKey>`

******************
Molecular Dynamics
******************

**Integrator**

* Velocity Verlet

**Thermostats**

* :ref:`Langevin <thermostatKey>`
* :ref:`Berendsen <thermostatKey>`
* :ref:`stochastic velocity rescaling <thermostatKey>`
* :ref:`Nose-Hoover chain <thermostatKey>`

**Manostats**

* :ref:`Berendsen <manostatKey>`
* :ref:`stochastic cell rescaling <manostatKey>`

**Cell coupling modes**

* :ref:`isotropic <isotropyKey>`
* :ref:`semi-isotropic <isotropyKey>`
* :ref:`anisotropic cell lengths <isotropyKey>`
* :ref:`full anisotropic cell lengths and angles <isotropyKey>`

**Constraints**

* :ref:`SHAKE/RATTLE <shakeKey>`
* :ref:`M-SHAKE <shakeKey>`
* :ref:`distance constraints <distanceConstraintsKey>`

************
Optimization
************

Geometry optimization is available for molecular-mechanics calculations. The
supported :ref:`optimizers <optimizerKey>` are:

* steepest descent
* ADAM

***
MPI
***

MPI support is used for QM-RPMD, where individual ring-polymer beads can be
distributed across ranks. Force evaluation itself remains local to each rank,
except for external QM programs that provide their own parallel execution.

*************
Planned Items
*************

The following items are planned for future releases:

* hybrid QM/MM job type
* MM-RPMD
* Verlet-list force evaluation
* Ewald summation
* LINCS constraints
