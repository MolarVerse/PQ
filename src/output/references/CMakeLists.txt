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

target_compile_definitions(references
    PUBLIC
    _REFERENCES_PATH_=\"${CMAKE_BINARY_DIR}/include/output/references/referenceFiles\"
)

add_custom_command(TARGET references POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/include/output/references/referenceFiles $<TARGET_FILE_DIR:references>/referenceFiles)

install(TARGETS references
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)