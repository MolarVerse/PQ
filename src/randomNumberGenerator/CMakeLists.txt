add_library(randomNumberGenerator
    randomNumberGenerator.cpp
)

target_include_directories(randomNumberGenerator
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/randomNumberGenerator
)

target_link_libraries(randomNumberGenerator
    PUBLIC
    settings
)

install(TARGETS randomNumberGenerator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)