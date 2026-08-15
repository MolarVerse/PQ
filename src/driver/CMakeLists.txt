set(driver
    driver.cpp
)

add_library(driver
    ${driver}
)

target_include_directories(driver
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/driver
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/engine
)

target_link_libraries(driver
    PUBLIC
    engine
    input
    setup
)

install(TARGETS driver
    DESTINATION lib
)

target_precompile_headers(driver REUSE_FROM pq_pch)
