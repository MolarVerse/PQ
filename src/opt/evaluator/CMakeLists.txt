add_library(evaluator
    evaluator.cpp
    hessianBuilder.cpp

    mmEvaluator.cpp
)

target_include_directories(evaluator
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/evaluator
)

target_link_libraries(evaluator
    PUBLIC
    linearAlgebra
    potential
    constraints
    physicalData
    virial
    molsys
    forceField
    intraNonBonded
)

install(TARGETS evaluator
    DESTINATION lib
)
