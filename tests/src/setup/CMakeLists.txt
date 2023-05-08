add_executable(testBoxSection testBoxSection.cpp)
target_include_directories(testBoxSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testBoxSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(testBoxSection testBoxSection)

add_executable(testNoseHooverSection testNoseHooverSection.cpp)
target_include_directories(testNoseHooverSection PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup)
target_link_libraries(testNoseHooverSection PRIVATE setup gtest gtest_main gmock gmock_main)
add_test(testNoseHooverSection testNoseHooverSection)