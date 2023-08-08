add_library(linearAlgebra
    vector3d.cpp
    matrix.cpp
)

target_include_directories(linearAlgebra
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/src/linearAlgebra
)

install(TARGETS linearAlgebra
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)