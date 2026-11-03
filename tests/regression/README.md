# Regression fixtures

Regression cases are stored as text fixtures and packaged into zip files by
CMake. The CTest runner unpacks the zip into a temporary directory, executes the
real `PQ` binary, and validates the manifest expectations.

The fixture format keeps binary zip output out of the source tree while still
testing the same user-facing path: unpack an example zip, run the input file,
and verify expected output files and completion markers.

`input` may list several input files; they run one after another in the same
directory. `compare_last_frame` compares the last frame of two `.xyz`-style
outputs (for example `.xyz` or `.vel`) and fails if any coordinate differs by
more than `atol`. `mm-md-shake-mshake` uses this to check that SHAKE and
M-SHAKE give the same trajectory for rigid water.

References for the invariants covered by the regression suite:

- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.:
  periodic boundary conditions and the minimum-image convention.
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed.:
  conservative forces as the negative derivative of potential energy.
