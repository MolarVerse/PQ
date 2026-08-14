add_library(linearAlgebra
    stlVector.cpp

    vector3d.cpp
    matrix.cpp
    staticMatrix.cpp
)

target_include_directories(linearAlgebra
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/concepts
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(linearAlgebra
    PUBLIC
    exceptions
    Eigen3::Eigen
)

install(TARGETS linearAlgebra
    DESTINATION lib
)

target_precompile_headers(linearAlgebra REUSE_FROM pq_pch)
