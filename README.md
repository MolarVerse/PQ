# PIMD-QMCF

[![C/C++ CI](https://github.com/97gamjak/pimd_qmcf/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/97gamjak/pimd_qmcf/actions/workflows/c-cpp.yml)
[![codecov](https://codecov.io/gh/97gamjak/pimd_qmcf/branch/main/graph/badge.svg?token=5WERM83FI0)](https://codecov.io/gh/97gamjak/pimd_qmcf)
[![Docs](https://github.com/97gamjak/pimd_qmcf/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://97gamjak.github.io/pimd_qmcf/)

## How to Use

To perform calculations using the PIMD-QMCF program just execute the executable `pimd_qmcf` with a given input file

    <path to executable>/pimd_qmcf <input file>

## Building from Source

Create a build directory and navigate into this directory. Within this directory configure cmake:

    cmake ../ -DCMAKE_BUILD_TYPE=Release

    make -j<#procs>

## Singularity

The `scripts` directory contains a file called `pimd_qmcf.def`, which is a definition file for a singularity container. First build this container as follows

    singularity build --fakeroot <name of container>.sif pimd_qmcf2.def

In order to run the application which is build within the container just execute

    singularity run --no-home <name of container>.sif <input file>








