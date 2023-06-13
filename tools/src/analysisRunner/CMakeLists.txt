add_library(analysisRunnerTools
    analysisRunner.cpp
    trajToCom.cpp
)

target_include_directories(analysisRunnerTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/analysisRunner
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(analysisRunnerTools
    PUBLIC
    frameTools
    configurationReaderTools
)