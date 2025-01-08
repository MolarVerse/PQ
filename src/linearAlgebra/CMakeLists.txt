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
    # SYSTEM ${eigen_SOURCE_DIR}
)

target_link_libraries(linearAlgebra
    PUBLIC
    exceptions
    Eigen3::Eigen
)

# quite ugly hack to avoid warnings when compiling Eigen with nvc++
get_target_property(Eigen_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
if(Eigen_INCLUDE_DIRS)
    target_include_directories(linearAlgebra SYSTEM PUBLIC ${Eigen_INCLUDE_DIRS})
endif()

install(TARGETS linearAlgebra
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)