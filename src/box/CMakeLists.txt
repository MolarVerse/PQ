add_library(box
    box.cpp
    orthorhombicBox.cpp
    triclinicBox.cpp
)

target_include_directories(box
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/box
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/device
)

target_link_libraries(box
    PRIVATE
    linearAlgebra
    exceptions
    settings
)

if(BUILD_WITH_GPU)
    target_link_libraries(box
        PRIVATE
        device
    )
endif()

install(TARGETS box
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)