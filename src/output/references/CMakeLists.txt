add_library(references
    referencesOutput.cpp
)

target_include_directories(references
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/output/references
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(references
    PUBLIC
    settings
)

install(TARGETS references
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)