# PQ

[![C/C++ CI](https://github.com/MolarVerse/PQ/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/MolarVerse/PQ/actions/workflows/c-cpp.yml)
[![codecov](https://codecov.io/gh/MolarVerse/PQ/branch/main/graph/badge.svg?token=5WERM83FI0)](https://codecov.io/gh/MolarVerse/PQ)
[![Docs](https://github.com/MolarVerse/PQ/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://MolarVerse.github.io/PQ/)

## Prerequisite

- CMake >= 3.12
- GCC   >= 13.0 

## How to Use

To perform calculations using the PQ program just execute the executable `PQ` with a given input file

    <path to executable>/PQ <input file>

## Building from Source

Create a build directory and navigate into this directory.

    mkdir build
    cd build

Within this directory configure cmake:

    cmake ../ -DCMAKE_BUILD_TYPE=Release

Optionally it is also possible to enable MPI for Ring Polymer MD

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=On

For compilation then type:

    make -j<#procs>

## Singularity

There are several singularity definition files shipped with this software package. For further information please refer to the [documentation page](https://MolarVerse.github.io/PQ/).








