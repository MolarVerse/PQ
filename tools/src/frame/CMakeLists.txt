add_library(frameTools
    atom.cpp
    molecule.cpp
    frame.cpp
)

target_include_directories(frameTools
    PUBLIC
    ${PROJECT_SOURCE_DIR}/tools/include/frame
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(frameTools
    PUBLIC
    linearAlgebra
    utilities
)

install(TARGETS frameTools
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/tools/
)