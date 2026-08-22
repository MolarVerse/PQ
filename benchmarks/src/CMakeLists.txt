function(add_pq_benchmark target source)
    add_executable(${target} ${source})
    target_include_directories(${target}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${target}
        PUBLIC
        simulationBox
        PRIVATE
        benchmark::benchmark_main
        ${ARGN}
    )

    add_test(
        NAME ${target}
        COMMAND ${target} --benchmark_min_time=0.001s
    )
    set_tests_properties(${target}
        PROPERTIES
        LABELS benchmark
        TIMEOUT 120
    )
endfunction()

add_pq_benchmark(
    benchmark_linearAlgebra
    math/benchmarkLinearAlgebra.cpp
    linearAlgebra
)
add_pq_benchmark(
    benchmark_box
    box/benchmarkBox.cpp
    box
    linearAlgebra
)
add_pq_benchmark(
    benchmark_pairPotentials
    potential/benchmarkPairPotentials.cpp
    coulombPotential
    nonCoulombPotential
)
add_pq_benchmark(
    benchmark_cellList
    simulationBox/benchmarkCellList.cpp
    simulationBox
)
add_pq_benchmark(
    benchmark_forceCalculation
    potential/benchmarkForceCalculation.cpp
    potential
)
add_pq_benchmark(
    benchmark_integrator
    integrator/benchmarkVelocityVerlet.cpp
    integrator
)
add_pq_benchmark(
    benchmark_kinetics
    simulationBox/benchmarkKinetics.cpp
    simulationBox
)

add_custom_target(google_benchmarks)
add_dependencies(
    google_benchmarks
    benchmark_linearAlgebra
    benchmark_box
    benchmark_pairPotentials
    benchmark_cellList
    benchmark_forceCalculation
    benchmark_integrator
    benchmark_kinetics
)
