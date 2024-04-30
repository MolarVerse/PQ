find_package(Python REQUIRED COMPONENTS Interpreter Development)

# check if pqanalysis is already installed if yes then skip the installation
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import PQAnalysis"
    RESULT_VARIABLE PQANALYSIS_INSTALLED
    OUTPUT_QUIET
)

if(PQANALYSIS_INSTALLED EQUAL 0)
    message(STATUS "PQAnalysis python package is already installed")
    return()
endif()

# check python version at least 3.12 in cmake
if(Python_VERSION VERSION_LESS 3.12)
    message(FATAL_ERROR "In order to build this project with python bindings you need at least Python 3.12")
endif()

# pip install pqanalysis
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pip install pqanalysis
    RESULT_VARIABLE PIP_INSTALL_RESULT
)

if(NOT PIP_INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install the PQAnalysis python package")
endif()