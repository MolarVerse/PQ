add_library(utilities
    executablePath.cpp
    stringUtilities.cpp
    mathUtilities.cpp
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
