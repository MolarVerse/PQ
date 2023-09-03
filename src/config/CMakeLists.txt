add_library(config
    commandLineArgs.cpp
)

target_include_directories(config
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(config
    PUBLIC
    exceptions
)

install(TARGETS config
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)