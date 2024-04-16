find_package(
    Python3
    REQUIRED
    COMPONENTS Interpreter
)

execute_process(
    COMMAND pip show gcovr
    RESULT_VARIABLE EXIT_CODE
    OUTPUT_QUIET
)

if(NOT ${EXIT_CODE} EQUAL 0 AND ${BUILD_WITH_GCOVR})
    message(
        FATAL_ERROR
        "The \"gcovr\" package is not installed. Please install it using the following command: \"pip3 install gcovr\"."
    )
endif()