add_library(box
    box.cpp
)

target_include_directories(box
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/box
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(box
    PUBLIC
    linearAlgebra
    exceptions
)

install(TARGETS box
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)