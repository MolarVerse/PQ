# Changelog

All notable changes to this project will be documented in this file.

## Next Release

### Enhancements

- Nose-Hoover chain restarting now including old chain parameters
- 'dftb_file' keyword added to change default input file dtfb.template
  for dftbplus QMMD
- input keys in input file can now be given case-insensitive as well as with '-' or '_'
- Checks for `CHANGELOG.md` modifications on pull requests and pulls

### Bug Fixes

- full anistrop coupling works now with stochastic cell rescaling manostat
  
<!-- insertion marker -->
## [v0.4.5](https://github.com/MolarVerse/PQ/releases/tag/v0.4.5) - 2024-07-13

### Bug Fixes

- Minimal Image Convention for triclinic cells now implemented with analytic extension
  
## [v0.4.4](https://github.com/MolarVerse/PQ/releases/tag/v0.4.4) - 2024-07-09

### Bug Fixes

- Anisotropic NPT calculations now working correctly

### Known Bugs

- Minimal Image Convention for triclinic cells only approximate

## [v0.4.3](https://github.com/MolarVerse/PQ/releases/tag/v0.4.3) - 2024-07-08

### Bug Fixes

- MACE NPT calculations bug fix - virial evaluation is now correct

### Known Bugs

- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.2](https://github.com/MolarVerse/PQ/releases/tag/v0.4.2) - 2024-07-04

### Bug Fixes

- Isotropic manostats producing SEGFAULTS is now fixed
- Version number in output files is now always the latest tag

### Testing

-Integration Test added for an exemplary NPT calculation using Berendsen-Thermostat and -Manostat (isotropic)

### Known Bugs

- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.1](https://github.com/MolarVerse/PQ/releases/tag/v0.4.1) - 2024-07-02

### Enhancements

- Logfile output updated to give all important information about the simulation settings

### CI

- added CI workflow for Kokkos enabled compilations

### Known Bugs

- Isotropic manostats producing SEGFAULTS
- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.0](https://github.com/MolarVerse/PQ/releases/tag/v0.4.0) - 2024-07-01

### Features

- M-Shake
- MACE Neural Network Potential for QM-MD calculations
- Steepest-Descent Optimizer and ADAM optimizer

### Known Bugs

- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate
