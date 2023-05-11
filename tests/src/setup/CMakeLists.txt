add_executable(testBoxSection testBoxSection.cpp)
target_include_directories(testBoxSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testBoxSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(NAME testBoxSection COMMAND testBoxSection WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

add_executable(testNoseHooverSection testNoseHooverSection.cpp)
target_include_directories(testNoseHooverSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testNoseHooverSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(NAME testNoseHooverSection COMMAND testNoseHooverSection WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

add_executable(testStepCountSection testStepCountSection.cpp)
target_include_directories(testStepCountSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testStepCountSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(NAME testStepCountSection COMMAND testStepCountSection WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)

add_executable(testAtomSection testAtomSection.cpp)
target_include_directories(testAtomSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testAtomSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(NAME testAtomSection COMMAND testAtomSection WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)