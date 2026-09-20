find_package(Doxygen)

if(DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    set(DOXYGEN_WARN_LOGFILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen_warnings.log)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND}
                -DDOXYGEN_EXECUTABLE=${DOXYGEN_EXECUTABLE}
                -DDOXYGEN_OUT=${DOXYGEN_OUT}
                -DWARN_LOGFILE=${DOXYGEN_WARN_LOGFILE}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/RunDoxygenWithLog.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else(DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")

    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E echo
                "Doxygen was not found at configure time - cannot build the 'docs' target. Install doxygen and re-run cmake."
        COMMAND ${CMAKE_COMMAND} -E false
        COMMENT "Doxygen not found"
        VERBATIM)
endif(DOXYGEN_FOUND)

find_program(SPHINX_EXECUTABLE NAMES sphinx-build)

if(DOXYGEN_FOUND AND SPHINX_EXECUTABLE)
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
    set(SPHINX_BUILD  ${SPHINX_SOURCE}/_build)

    set(SPHINX_HTML_STAMP ${SPHINX_BUILD}/.sphinx-html-stamp)

    add_custom_command(
        OUTPUT ${SPHINX_HTML_STAMP}
        COMMAND ${CMAKE_COMMAND} -E env PQ_DOXYGEN_XML_DIR=${CMAKE_CURRENT_BINARY_DIR}/xml
                ${SPHINX_EXECUTABLE} -b html . _build/html
        COMMAND ${CMAKE_COMMAND} -E make_directory _build/html/doxygen
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/html _build/html/doxygen
        COMMAND ${CMAKE_COMMAND} -E touch ${SPHINX_HTML_STAMP}
        WORKING_DIRECTORY ${SPHINX_SOURCE}
        COMMENT "Building Sphinx HTML site"
        VERBATIM
    )

    add_custom_target(sphinx-docs DEPENDS ${SPHINX_HTML_STAMP})
    add_dependencies(sphinx-docs docs)
else()
    message("Doxygen and Sphinx (sphinx-build) are both required to build the Sphinx documentation")

    add_custom_target(sphinx-docs
        COMMAND ${CMAKE_COMMAND} -E echo
                "Doxygen and/or Sphinx (sphinx-build) were not found at configure time - cannot build the 'sphinx-docs' target. Install both and re-run cmake."
        COMMAND ${CMAKE_COMMAND} -E false
        COMMENT "Doxygen and/or Sphinx not found"
        VERBATIM)
endif()
