set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/tests )

add_executable(testRstFileSection src/testRstFileSection.cpp)
target_include_directories(testRstFileSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include)
target_link_libraries(testRstFileSection PRIVATE setup PRIVATE gtest PRIVATE gtest_main PRIVATE gmock PRIVATE gmock_main) # TODO: check if gmock inclusion works also at home
add_test(NAME testRstFileSection COMMAND testRstFileSection WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})