add_library(eningeTools
    engine.cpp
)

target_include_directories(eningeTools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/include/engine
    ${PROJECT_SOURCE_DIR}/external/progressbar/include
)

target_link_libraries(eningeTools
    PUBLIC
)