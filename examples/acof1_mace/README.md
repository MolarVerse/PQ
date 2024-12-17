# MACE: Example with ACOF1

## Description

This example shows how [MACE](https://github.com/ACEsuit/mace) can be used to simulate solid-state systems, in this case a trigonal covalent organic framework (ACOF1). 

## Input files

The input files for this example are:

- `acof1.rst`: The atomic structure of the ACOF1 framework in the `.rst` format.
- `moldescriptor.dat`: The molecular descriptor file for the ACOF1 framework.
- `run-01.in`: The input file for the MACE simulation.

## Settings in `run-01.in`

The input file `run-01.in` contains the following settings:

- `jobtype = qm-md`: QM-MD simulation.
- `init_velocities = true`: Boltzmann distribution of velocities.
- `thermostat = velocity-rescaling`: Velocity rescaling thermostat (Bussi).
- `manostat = stochastic_rescaling`: Stochastic rescaling manostat (Bussi).
- `isotropy = full_anisotropic`: Full anisotropic pressure coupling.
- `qm_prog = mace_mp`: MACE-MP-0 model for the "QM" calculation.
- `mace_model_size = large`: Large model size.
- `dispersion = true`: Dispersion correction on.
- `floating_point_type = float`: Single precision.