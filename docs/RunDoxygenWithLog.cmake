execute_process(
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    RESULT_VARIABLE DOXYGEN_RESULT
)

if(EXISTS ${WARN_LOGFILE})
    file(READ ${WARN_LOGFILE} DOXYGEN_WARNINGS)
    if(NOT "${DOXYGEN_WARNINGS}" STREQUAL "")
        message("${DOXYGEN_WARNINGS}")
    endif()
endif()

if(NOT DOXYGEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Doxygen failed with exit code ${DOXYGEN_RESULT}")
endif()
