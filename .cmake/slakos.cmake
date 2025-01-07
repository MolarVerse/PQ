include(FetchContent)

function(CloneRepository repositoryURL sourceDir)
    #Commands are left empty so that we only checkout the source and no not perform any kind of build
    if (EXISTS ${sourceDir})
        message("Directory ${sourceDir} already exists. Skipping cloning of ${repositoryURL}")
        return()
    endif()

    message("Starting to clone ${repositoryURL} into ${sourceDir}")
    execute_process(
        COMMAND git clone ${repositoryURL} ${sourceDir}
        RESULT_VARIABLE result
    )

    if(NOT ${result} EQUAL 0)
        message(FATAL_ERROR "Failed to clone from ${repositoryURL}")
    endif()
endfunction(CloneRepository)

# fetch 3ob slakos files
CloneRepository("https://github.com/dftbparams/3ob.git" "${CMAKE_BINARY_DIR}/external/slakos/3ob")

# fetch matsci files
CloneRepository("https://github.com/dftbparams/matsci.git" "${CMAKE_BINARY_DIR}/external/slakos/matsci")

# define directory for 3ob and matsci for preprocessor
add_compile_definitions(SLAKOS_DIR="${CMAKE_BINARY_DIR}/external/slakos")
