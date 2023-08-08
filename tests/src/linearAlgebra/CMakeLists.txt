set(source_files
    testVector3d.cpp
    testMatrix.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})

    # target_include_directories(${test_name} PRIVATE)
    target_link_libraries(${test_name} PRIVATE linearAlgebra gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()