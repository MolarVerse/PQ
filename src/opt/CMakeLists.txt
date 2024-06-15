add_library(optimization
    opt.cpp
)

target_include_directories(optimization
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/learningRateStrategy
    ${PROJECT_SOURCE_DIR}/include/opt/evaluator
)

target_link_libraries(optimization
    PUBLIC
    learningRateStrategy
    optimizer
    evaluator
)

install(TARGETS optimization
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(learningRateStrategy)
add_subdirectory(optimizer)
add_subdirectory(evaluator)
