add_library(linearAlgebra
    stlVector.cpp

    vector3d.cpp
    matrix.cpp
    staticMatrix3x3.cpp
)

target_include_directories(linearAlgebra
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/src/linearAlgebra
)

target_link_libraries(linearAlgebra
    PUBLIC
)

install(TARGETS linearAlgebra
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)