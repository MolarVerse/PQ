add_library(commandLineArgs
    commandLineArgs.cpp
)

target_include_directories(commandLineArgs
    PUBLIC
    ${PROJECT_SOURCE_DIR}/apps/commandLineArgs.hpp
)

target_link_libraries(commandLineArgs
    PRIVATE
    exceptions
)

install(TARGETS commandLineArgs
    DESTINATION bin
)
