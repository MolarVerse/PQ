# PQ

[![BUILD](https://github.com/MolarVerse/PQ/actions/workflows/ci_build.yml/badge.svg)](https://github.com/MolarVerse/PQ/actions/workflows/ci_build.yml)
[![codecov](https://codecov.io/gh/MolarVerse/PQ/branch/main/graph/badge.svg?token=5WERM83FI0)](https://codecov.io/gh/MolarVerse/PQ)
[![Docs](https://github.com/MolarVerse/PQ/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://MolarVerse.github.io/PQ/)

## About

PQ is an open-source platform for advanced molecular dynamics simulations, offering comprehensive support for both classical molecular mechanics (MM) and quantum mechanics (QM) calculations.

Key features include:

- **Multiple simulation engines**: Support for MM-MD, QM-MD, QM-RPMD and planned QM/MM-MD calculations
- **Quantum mechanics integration**: Compatible with DFTB+, Turbomole, PySCF and MACE engines
- **Flexible force fields**: GUFF (Grand Unified Force Field) and AMBER-type force fields with various non-bonded interaction models (Lennard-Jones, Buckingham, Morse)
- **Advanced algorithms**: Ring polymer molecular dynamics for quantum nuclear effects, various thermostats and manostats
- **High-performance computing**: MPI support for parallel simulations and optional Kokkos acceleration
- **Integrated ecosystem**: Part of the [MolarVerse](https://github.com/MolarVerse) organization, providing seamless integration with companion tools for trajectory analysis, structure preparation and CLI utilities

PQ is designed for researchers in computational chemistry and materials science who require accurate and efficient molecular dynamics simulations across different scales and methodologies.

## Documentation

Comprehensive documentation is available on the [PQ documentation website](https://MolarVerse.github.io/PQ/), covering:

- **User Guide**: Detailed instructions on input file structure, necessary setup files and generated output files
- **Installation Guide**: Step-by-step building instructions, including Singularity containers
- **Feature List**: Complete overview of implemented features
- **Developer Guide**: Project architecture, software testing and contributing guidelines

## Examples

Multiple examples of different chemical systems and jobtypes are given in the `examples` directory.

## How to Use

To perform calculations using the PQ program, execute the executable `PQ` with a given input file

    <path to executable>/PQ <input file>

## Building from Source

Prerequisites:

- CMake >= 3.18
- GCC   >= 13.0

Clone the PQ GitHub repository and navigate into the directory:

    git clone https://github.com/MolarVerse/PQ.git
    cd PQ

Create a build directory and navigate into this directory:

    mkdir build
    cd build

Within this directory configure CMake:

    cmake ../ -DCMAKE_BUILD_TYPE=Release

Optionally it is also possible to enable MPI for Ring Polymer MD

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For compilation then type:

    make -j<#procs>

The executable binary is then found in the subfolder named "apps" inside the build directory.

## Citation

If you use PQ in your research, please cite:

- Gamper, J., Gallmetzer, J. M., Weiss, A. K. H., & Hofer, T. S. (2025). PQ: An Open-Source Platform for Advanced Molecular Dynamics Simulations. Zenodo. <https://doi.org/10.5281/zenodo.14185071>

- Gamper, J., Gallmetzer, J. M., Weiss, A. K. H., & Hofer, T. S. (2024). PQAnalysis (1.2.1). Zenodo. <https://doi.org/10.5281/zenodo.11322103>