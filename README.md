# PIMD-QMCF

[![C/C++ CI](https://github.com/pimd-qmcf/pimd_qmcf/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/pimd-qmcf/pimd_qmcf/actions/workflows/c-cpp.yml)
[![codecov](https://codecov.io/gh/pimd-qmcf/pimd_qmcf/branch/main/graph/badge.svg?token=5WERM83FI0)](https://codecov.io/gh/pimd-qmcf/pimd_qmcf)
[![Docs](https://github.com/pimd-qmcf/pimd_qmcf/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://pimd-qmcf.github.io/pimd_qmcf/)

## How to Use

To perform calculations using the PIMD-QMCF program just execute the executable `pimd_qmcf` with a given input file

    <path to executable>/pimd_qmcf <input file>

## Building from Source

Create a build directory and navigate into this directory. Within this directory configure cmake:

    cmake ../ -DCMAKE_BUILD_TYPE=Release

Optionally it is also possible to enable MPI for Ring Polymer MD

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For compilation then type:

    make -j<#procs>

## Singularity

There are several singularity definition files shipped with this software package. For further information please refer to the [documentation page](https://pimd-qmcf.github.io/pimd_qmcf/).








