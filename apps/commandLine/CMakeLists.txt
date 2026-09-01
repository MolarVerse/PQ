add_library(commandLineArgs
    commandLineArgs.cpp
)

target_include_directories(commandLineArgs
    PUBLIC
    ${PROJECT_SOURCE_DIR}/apps/commandLine
)

target_link_libraries(commandLineArgs
    PRIVATE
    exceptions
)

install(TARGETS commandLineArgs
    DESTINATION lib
)
