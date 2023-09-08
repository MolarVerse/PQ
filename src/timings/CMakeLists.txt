add_library(timings
    timings.cpp
)

target_include_directories(timings
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/timings
)

install(TARGETS timings
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)