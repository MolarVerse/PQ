# MACE: Example with H2O

## Description

This example shows how [MACE](https://github.com/ACEsuit/mace) can be used to simulate water (H2O) molecules in NVE ensemble.

## Input files

The input files for this example are:

- [`h2o.rst`](h2o.rst): The atomic structure of the ten water molecules in the `.rst` format.
- [`moldescriptor.dat`](moldescriptor.dat): The molecular descriptor file for the water molecules.
- [`run-01.in`](run-01.in): The input file for the MACE simulation.

## Settings in [`run-01.in`](run-01.in)

The input file `run-01.in` contains the following settings:

- `jobtype = qm-md`: QM-MD simulation.
- `qm_prog = mace_mp`: MACE-MP-0 model for the "QM" calculation.
- `mace_model_size = medium`: Large model size.
- `floating_point_type = float`: Single precision.