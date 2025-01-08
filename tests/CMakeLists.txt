# suppress warnings for GTests for nvc++ compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    add_compile_options(
        "--diag_suppress code_is_unreachable"
    )
endif()

add_subdirectory(src)