add_library(math
    vector3d.cpp
)

target_include_directories(math
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/math
)

target_link_libraries(math
    PUBLIC
)

install(TARGETS math
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)