add_library(pq_test_main
    main.cpp
)

target_link_libraries(pq_test_main
    PUBLIC
    gtest
    gmock
)

install(TARGETS pq_test_main
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
