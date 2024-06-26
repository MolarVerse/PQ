add_library(learningRateStrategy
    learningRateStrategy.cpp

    constant.cpp
    constantDecay.cpp
    expDecay.cpp
)

target_include_directories(learningRateStrategy
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/learningRateStrategy
)

target_link_libraries(learningRateStrategy
    PUBLIC
    linearAlgebra
    settings
)

install(TARGETS learningRateStrategy
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)