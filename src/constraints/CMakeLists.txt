add_library(constraints
    constraints.cpp
    bondConstraint.cpp
)

target_include_directories(constraints
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/constraints
)

target_link_libraries(constraints
    PUBLIC
    simulationBox
)

install(TARGETS constraints
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)