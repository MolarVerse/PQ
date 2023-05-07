add_library(setup
    rstFileReader.cpp
    rstFileSection.cpp
    inputFileReader.cpp
    inputFileParser.cpp    
    inputFileParserOutput.cpp 
    inputFileParserJobtype.cpp
    inputFileParserTimings.cpp
    inputFileParserGeneral.cpp
    inputFileParserIntegrator.cpp
    moldescriptorReader.cpp
    postProcessSetup.cpp
)

target_include_directories(setup
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
        engine
        utilities
        output
        config
)