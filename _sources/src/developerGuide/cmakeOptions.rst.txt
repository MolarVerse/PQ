.. _buildOptions:

#############
Build Options
#############

The options below are defined in ``.cmake/config.cmake``. Pass them to the first ``cmake`` call (``-DNAME=VALUE``); the
values are cached in the build directory.

.. list-table::
    :header-rows: 1
    :widths: 32 10 58

    * - Option
      - Default
      - Meaning
    * - ``BUILD_SHARED_LIBS``
      - ON
      - Build the modules as shared libraries. Most of CI uses it; the static builds are separate jobs.
    * - ``BUILD_WITH_TESTS``
      - ON
      - Build the GoogleTest unit tests (``ctest``).
    * - ``BUILD_WITH_MPI``
      - OFF
      - MPI support, needed for ring polymer MD. CI has its own ``mpi-build`` job.
    * - ``BUILD_WITH_PYTHON_BINDINGS``
      - OFF
      - Python bindings. CI builds them in the ``python-bindings`` job and runs ``testPQCli``.
    * - ``BUILD_WITH_IWYU``
      - OFF
      - Run include-what-you-use during the build.
    * - ``BUILD_WITH_GCOVR``
      - OFF
      - Coverage with ``gcovr`` (the Debug job on x86_64 uses it).
    * - ``BUILD_WITH_DOCS``
      - ON
      - The documentation targets (``sphinx-docs``, ``docs``). Turn it off if you only build the code.
    * - ``PQ_DOXYGEN_GRAPHS``
      - ON
      - Draw the dot graphs in the Doxygen output. Off in the pull request builds because it is slow.
    * - ``BUILD_WITH_BENCHMARKING``
      - OFF
      - The Google Benchmark suite in ``benchmarks/``.
    * - ``BUILD_WITH_ASE``
      - ON
      - ASE integration, which the QM runner programs use.
    * - ``BUILD_WITH_SINGULARITY``
      - OFF
      - Singularity container build.
    * - ``BUILD_WITH_LTO``
      - OFF
      - Link-time optimisation for Release builds.
    * - ``BUILD_WITH_NATIVE``
      - ON
      - ``-march=native`` for Release. Turn it **off** for anything that runs on another machine: CI, containers,
        shared binaries.
    * - ``BUILD_WITH_PERF_BENCH``
      - OFF
      - The fixed-work benchmarks in ``benchmarks/perf`` that the Performance Gate runs under ``callgrind``.

Two things are wired in automatically when they are found on the ``PATH``: ``ccache`` as compiler launcher and
``mold`` or ``ld.lld`` as linker. A build that is much slower than usual often has neither; check the start of the
CMake output.

Typical combinations:

.. code:: bash

    # what the CI Debug job does (strict warnings with -Werror)
    $ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_NATIVE=OFF

    # fast local Release build without the documentation targets
    $ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_DOCS=OFF

    # compile database for clangd and clangd-tidy
    $ cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

The ``Debug`` build type enables the strict warning set described on :doc:`staticChecks`. A Release build does not, so a
warning can show up for the first time in CI.
