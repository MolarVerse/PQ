add_executable(testSelection
    testSelection.cpp
)

target_link_libraries(testSelection
    PRIVATE
    pq_python
    gtest
    pq_test_main
)

add_test_with_gcda_cleanup(testSelection testSelection)

set_property(TEST testSelection PROPERTY LABELS python)
