find_package(Python REQUIRED COMPONENTS Interpreter Development)

# check if python version is at least 3.12
if(Python_VERSION VERSION_LESS 3.12)
    message(FATAL_ERROR "Python version 3.12 or higher is required")
endif()

# check if pqanalysis is already installed if yes then skip the installation
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import PQAnalysis"
    RESULT_VARIABLE PQANALYSIS_INSTALLED
    OUTPUT_QUIET
)

if(PQANALYSIS_INSTALLED EQUAL 1)
    execute_process(

        # TODO: as soon as next release is out, change this to <pip install pqanalysis>
        COMMAND ${Python_EXECUTABLE} -m pip install update PQAnalysis
        RESULT_VARIABLE PIP_INSTALL_RESULT
    )
    set(PQANALYSIS_INSTALLED PIP_INSTALL_RESULT)
endif()

if(NOT PQANALYSIS_INSTALLED EQUAL 0)
    message(FATAL_ERROR "Failed to install the PQAnalysis python package")
else()
    message(STATUS "Successfully installed the PQAnalysis python package")
endif()

# check if pqanalysis is already installed if yes then skip the installation
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11"
    RESULT_VARIABLE PYBIND11_INSTALLED
    OUTPUT_QUIET
)