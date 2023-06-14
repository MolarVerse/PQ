add_library(configurationReaderTools
    configurationReader.cpp
    extxyzReader.cpp
)

target_include_directories(configurationReaderTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/configurationReader
)

target_link_libraries(configurationReaderTools
    PUBLIC
    frameTools
    math
)

install(TARGETS configurationReaderTools
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/tools/
)