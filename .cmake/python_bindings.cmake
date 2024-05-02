find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

# check if pqanalysis is already installed if yes then skip the installation
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import PQAnalysis"
    RESULT_VARIABLE PQANALYSIS_INSTALLED
    OUTPUT_QUIET
)

if(PQANALYSIS_INSTALLED EQUAL 1)
    # pip install pqanalysis
    execute_process(

        # TODO: as soon as next release is out, change this to <pip install pqanalysis>
        COMMAND ${Python_EXECUTABLE} -m pip install git+https://github.com/MolarVerse/PQAnalysis.git@dev
        RESULT_VARIABLE PIP_INSTALL_RESULT
    )

    if(NOT PIP_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install the PQAnalysis python package")
    endif()
endif()