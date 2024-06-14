add_library(optimizer
    optimizer.cpp
)

target_include_directories(optimizer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/optimizer
)

target_link_libraries(optimizer
    PUBLIC
    linearAlgebra
)

install(TARGETS optimizer
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)