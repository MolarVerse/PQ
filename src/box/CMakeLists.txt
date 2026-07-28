add_library(box
    box.cpp
    orthorhombicBox.cpp
    triclinicBox.cpp
)

target_include_directories(box
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/box
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(box
    PRIVATE
    linearAlgebra
    exceptions
    settings
)

install(TARGETS box
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)