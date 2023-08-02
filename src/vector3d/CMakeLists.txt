add_library(vector3d
    vector3d.cpp
)

target_include_directories(vector3d
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/vector3d
)

install(TARGETS vector3d
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)