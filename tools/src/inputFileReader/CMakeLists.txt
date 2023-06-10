add_library(inputFileReaderTools
    inputFileReader.cpp
    trajToComInFileReader.cpp
)

target_include_directories(inputFileReaderTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/inputFileReader
    ${PROJECT_SOURCE_DIR}/external/toml++/include
)

target_link_libraries(inputFileReaderTools
    PUBLIC
    analysisRunnerTools
)