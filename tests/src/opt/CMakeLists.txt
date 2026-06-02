set(source_files
    testLearningRateStrategy.cpp
    testConvergence.cpp
    testOptimizer.cpp
    testSteepestDescent.cpp
    testAdam.cpp
    testMMEvaluator.cpp
)

# Per-source link/include overrides so each test only pulls in the libraries
# it actually needs.
set(testLearningRateStrategy_includes
    ${PROJECT_SOURCE_DIR}/include/opt/learningRateStrategy
)
set(testLearningRateStrategy_libs
    learningRateStrategy
)

set(testConvergence_includes
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/settings
    ${PROJECT_SOURCE_DIR}/include/config
)
set(testConvergence_libs
    convergence
    settings
)

set(testOptimizer_includes
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/settings
    ${PROJECT_SOURCE_DIR}/include/exceptions
    ${PROJECT_SOURCE_DIR}/include/physicalData
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra/vector3d
    ${PROJECT_SOURCE_DIR}/include/config
)
set(testOptimizer_libs
    optimizer
    convergence
    physicalData
    simulationBox
    exceptions
    settings
    linearAlgebra
)

set(testSteepestDescent_includes
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra/vector3d
    ${PROJECT_SOURCE_DIR}/include/config
)
set(testSteepestDescent_libs
    optimizer
    convergence
    simulationBox
    linearAlgebra
)

set(testAdam_includes
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra/vector3d
    ${PROJECT_SOURCE_DIR}/include/config
)
set(testAdam_libs
    optimizer
    convergence
    simulationBox
    linearAlgebra
)

set(testMMEvaluator_includes
    ${PROJECT_SOURCE_DIR}/include/opt/evaluator
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/physicalData
    ${PROJECT_SOURCE_DIR}/include/potential
    ${PROJECT_SOURCE_DIR}/include/forceField
    ${PROJECT_SOURCE_DIR}/include/intraNonBonded
    ${PROJECT_SOURCE_DIR}/include/virial
    ${PROJECT_SOURCE_DIR}/include/constraints
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra/vector3d
    ${PROJECT_SOURCE_DIR}/include/config
)
set(testMMEvaluator_libs
    evaluator
    simulationBox
    physicalData
    potential
    forceField
    intraNonBonded
    virial
    constraints
    linearAlgebra
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
        ${${test_name}_includes}
    )
    target_link_libraries(${test_name}
        PRIVATE
        ${${test_name}_libs}
        gtest
        gmock
        pq_test_main
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    set_property(TEST ${test_name} PROPERTY LABELS opt)
endforeach()
