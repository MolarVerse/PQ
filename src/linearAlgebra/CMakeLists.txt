add_library(linearAlgebra
    stlVector.cpp

    vector3d.cpp
    vector3dConcepts.cpp
    matrix.cpp
    staticMatrix.cpp
)

target_include_directories(linearAlgebra
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
    ${PROJECT_SOURCE_DIR}/include/concepts
)

target_link_libraries(linearAlgebra
    PUBLIC
    exceptions
    Eigen3::Eigen
    concepts
)

install(TARGETS linearAlgebra
    DESTINATION lib
)

target_precompile_headers(linearAlgebra REUSE_FROM pq_pch)

# vector3dConcepts.cpp exists only to give clangd/libclang tooling a real
# compile command for a header with no other matching translation unit; it
# needs none of the PCH's contents (<format>, <memory>, Eigen/Dense), so skip
# it to avoid Eigen's SSE intrinsics headers tripping up libclang-based
# checks (devops_cpp_check) that only configure, never build, the project.
set_source_files_properties(
    vector3dConcepts.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)
