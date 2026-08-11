add_library(timings
    timer.cpp
    timingsSection.cpp
    timingsSectionGuard.cpp
    globalTimer.cpp
)

target_include_directories(timings
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/timings
)

target_link_libraries(timings
    PUBLIC
    exceptions
    settings
)

install(TARGETS timings
    DESTINATION lib
)
