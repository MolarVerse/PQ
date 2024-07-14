# PQ

[![C/C++ CI](https://github.com/MolarVerse/PQ/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/MolarVerse/PQ/actions/workflows/c-cpp.yml)
[![codecov](https://codecov.io/gh/MolarVerse/PQ/branch/main/graph/badge.svg?token=5WERM83FI0)](https://codecov.io/gh/MolarVerse/PQ)
[![Docs](https://github.com/MolarVerse/PQ/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://MolarVerse.github.io/PQ/)


## How to Use

To perform calculations using the PQ program just execute the executable `PQ` with a given input file

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

## Singularity

There are several singularity definition files shipped with this software package. For further information please refer to the [documentation page](https://MolarVerse.github.io/PQ/).








