add_library(optimizer
    optimizer.cpp

    steepestDescent.cpp
)

target_include_directories(optimizer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
)

target_link_libraries(optimizer
    PRIVATE
    linearAlgebra
)

install(TARGETS optimizer
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)