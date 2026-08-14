add_library(pq_test_main
    main.cpp
)

target_link_libraries(pq_test_main
    PUBLIC
    gtest
    gmock
)
