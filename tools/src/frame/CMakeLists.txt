add_library(frameTools
    atom.cpp
    molecule.cpp
    frame.cpp
)

target_include_directories(frameTools
    PUBLIC
    ${PROJECT_SOURCE_DIR}/tools/include/frame
)

target_link_libraries(frameTools
    PUBLIC
    vector3d
    config
)

install(TARGETS frameTools
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/tools/
)