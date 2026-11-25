add_executable(testSelection
    testSelection.cpp
)

target_compile_definitions(testSelection
    PRIVATE
    WITH_PYTHON_BINDINGS
)

target_link_libraries(testSelection
    PRIVATE
    pq_python
    gtest
    pq_test_main
)

add_test_with_gcda_cleanup(testSelection testSelection)

set_property(TEST testSelection PROPERTY LABELS python)

add_executable(testPythonError
    testPythonError.cpp
)

target_compile_definitions(testPythonError
    PRIVATE
    WITH_PYTHON_BINDINGS
)

target_link_libraries(testPythonError
    PRIVATE
    pq_python
    gtest
    pq_test_main
)

add_test_with_gcda_cleanup(testPythonError testPythonError)

set_property(TEST testPythonError PROPERTY LABELS python)
