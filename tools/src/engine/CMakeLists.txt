add_library(engineTools
    engine.cpp
)

target_include_directories(engineTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/engine
    ${PROJECT_SOURCE_DIR}/external/progressbar/include
)

target_link_libraries(engineTools
    PUBLIC
    analysisRunnerTools
)