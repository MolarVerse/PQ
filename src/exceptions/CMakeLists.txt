add_library(exceptions
    exceptions.cpp
)

target_include_directories(exceptions
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/exceptions
)

install(TARGETS exceptions
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)