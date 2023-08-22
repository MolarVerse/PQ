set(source_files
    testForceField.cpp
    testAngleType.cpp
    testBondType.cpp
    testDihedralType.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/include/constraints ${PROJECT_SOURCE_DIR}/tests/include/macros)
    target_link_libraries(${test_name} PRIVATE forceField config gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()