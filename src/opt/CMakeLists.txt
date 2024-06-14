add_library(optimization
    opt.cpp
)

target_include_directories(optimization
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/learningRateStrategy
)

target_link_libraries(optimization
    PRIVATE
    learningRateStrategy
    optimizer
)

install(TARGETS optimization
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(learningRateStrategy)
add_subdirectory(optimizer)