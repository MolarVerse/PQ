add_library(box
    box.cpp
    orthorhombicBox.cpp
    triclinicBox.cpp
)

target_include_directories(box
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/box
)

target_link_libraries(box
    PUBLIC
    config_headers
    PRIVATE
    linearAlgebra
    exceptions
    settings
)

install(TARGETS box
    DESTINATION lib
)
