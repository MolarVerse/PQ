add_library(testUtils
    testUtils.cpp
)

target_include_directories(testUtils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tests/include/testUtils/
)

target_link_libraries(testUtils
    PUBLIC
    engine
)

install(TARGETS testUtils
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
