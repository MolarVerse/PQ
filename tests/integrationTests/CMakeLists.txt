find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Pytest REQUIRED)

set(TEST_NAME cgo_mmmd)
add_test(NAME ${TEST_NAME}
    COMMAND
    Pytest::Pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/testsets/test_${TEST_NAME}.py
    --executable ${CMAKE_BINARY_DIR}/apps/PQ
    --name ${TEST_NAME}
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)
