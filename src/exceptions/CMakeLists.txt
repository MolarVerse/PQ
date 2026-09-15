add_library(exceptions 
    baseException.cpp
)

target_include_directories(exceptions
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/exceptions
)

target_link_libraries(exceptions
    PUBLIC
    mstd
)

install(TARGETS exceptions
    DESTINATION lib
)
