# Each performance-regression benchmark is a standalone fixed-work executable.
# To add one: drop a <name>.cpp here and add it to the list below. The CI gate
# (.github/workflows/perf.yml) auto-discovers every perf_* target.
set(perf_benchmarks
    forceKernel.cpp
    coulombKernel.cpp
    nonCoulombPairs.cpp
    shiftVector.cpp
    linearAlgebra.cpp
    boxTransforms.cpp
    bondedForces.cpp
    integrator.cpp
    kinetics.cpp
    virial.cpp
    constraints.cpp
    perfInterWater.cpp
)

include(CheckIncludeFileCXX)
check_include_file_cxx("valgrind/callgrind.h" HAVE_CALLGRIND_H)

set(perf_targets "")
foreach(perf_source ${perf_benchmarks})
    get_filename_component(perf_name ${perf_source} NAME_WE)
    add_executable(perf_${perf_name} ${perf_source})

    target_include_directories(perf_${perf_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(perf_${perf_name}
        PRIVATE
        intraNonBonded
        forceField
        integrator
        virial
        constraints
        simulationBox
        potential
        interWater
        settings
        physicalData
        box
        linearAlgebra
    )

    # enable the callgrind counter hook if valgrind's header is available
    if(HAVE_CALLGRIND_H)
        target_compile_definitions(perf_${perf_name} PRIVATE PQ_WITH_CALLGRIND)
    endif()

    list(APPEND perf_targets perf_${perf_name})
endforeach()

# aggregate target so CI can build every benchmark with one command
add_custom_target(perf_benchmarks)
add_dependencies(perf_benchmarks ${perf_targets})
