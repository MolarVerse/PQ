add_library(analysisTools
    analysis.cpp
    trajectoryToCom.cpp
)

target_include_directories(analysisTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/analysis
)

target_link_libraries(analysisTools
    PUBLIC
    frameTools
    configurationReaderTools
)