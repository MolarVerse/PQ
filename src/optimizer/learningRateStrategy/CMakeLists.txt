add_library(learningRateStrategy
    learningRateStrategy.cpp
)

target_include_directories(learningRateStrategy
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/optimizer/learningRateStrategy
)

target_link_libraries(learningRateStrategy
    PUBLIC
    linearAlgebra
)

install(TARGETS learningRateStrategy
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)