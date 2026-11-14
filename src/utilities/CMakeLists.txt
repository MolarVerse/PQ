add_library(utilities
    executablePath.cpp
    stringUtilities.cpp
    mathUtilities.cpp
    strongTypes.cpp
    systemMemory.cpp
)

target_include_directories(utilities
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/utilities
)

target_link_libraries(utilities
    PUBLIC
    linearAlgebra
    exceptions
    mstd
)

install(TARGETS utilities
    DESTINATION lib
)

find_package(Threads REQUIRED)

add_library(serialWorker
    serialWorker.cpp
)

target_include_directories(serialWorker
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/utilities
)

target_link_libraries(serialWorker
    PUBLIC
    Threads::Threads
)

install(TARGETS serialWorker
    DESTINATION lib
)

add_library(interrupt
    interrupt.cpp
)

target_include_directories(interrupt
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/utilities
)

if(BUILD_WITH_PYBIND11)
    target_link_libraries(interrupt
        PRIVATE
        pybind11::embed
    )
endif()

install(TARGETS interrupt
    DESTINATION lib
)
