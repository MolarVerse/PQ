add_library(frameTools
    atom.cpp
    molecule.cpp
    frame.cpp
)

target_include_directories(frameTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/frame
)

target_link_libraries(frameTools
    PUBLIC
    math
    config
)

install(TARGETS frameTools
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/tools/
)