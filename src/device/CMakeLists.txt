add_library(device
    device.cpp
)

target_include_directories(device
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/device
)

target_link_libraries(device
    PUBLIC
    exceptions
)

if(BUILD_WITH_GPU)
    target_link_libraries(device
        PUBLIC
        cuda
        cudart
    )
endif()

install(TARGETS device
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)