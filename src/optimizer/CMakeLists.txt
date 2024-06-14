add_library(optimizer
    optimizer.cpp

    steepestDescent.cpp
)

target_include_directories(optimizer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/optimizer
    ${PROJECT_SOURCE_DIR}/include/optimizer/learningRateStrategy
)

target_link_libraries(optimizer
    PRIVATE
    linearAlgebra

    PUBLIC
    learningRateStrategy
)

install(TARGETS optimizer
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(learningRateStrategy)