add_library(tomlExtensions
    tomlExtensions.cpp
)

target_include_directories(tomlExtensions
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/toml
    ${PROJECT_SOURCE_DIR}/external/toml++/include
)

target_link_libraries(tomlExtensions
    PUBLIC
)

install(TARGETS tomlExtensions
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/tools/
)