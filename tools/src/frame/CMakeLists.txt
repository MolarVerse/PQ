add_library(frameTools
    frame.cpp
)

target_include_directories(frameTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/frame
)

target_link_libraries(frameTools
    PUBLIC
    math
)