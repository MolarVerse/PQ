add_library(outputTools
    trajOutput.cpp
)

target_include_directories(outputTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/output
)

target_link_libraries(outputTools
    PUBLIC
    frameTools
    output
)