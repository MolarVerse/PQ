set(source_files
    minimalImageConvention.cpp
)

foreach(source_file ${source_files})
    get_filename_component(benchmark_name ${source_file} NAME_WE)
    add_executable(${benchmark_name} ${source_file})

    target_include_directories(${benchmark_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${benchmark_name} PRIVATE benchmark)
    add_test(NAME ${benchmark_name} COMMAND ${benchmark_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmarks)
endforeach()