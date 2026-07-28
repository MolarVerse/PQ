set(
    PQ_SLAKOS_SOURCE_DIR
    ""
    CACHE PATH
    "Path to preseeded 3ob and matsci directories for offline builds"
)

function(ValidateSlakosSet source_dir set_name)
    foreach(required_path IN ITEMS
        LICENSE
        README
        RELEASE.md
        CHANGELOG.md
    )
        if(NOT EXISTS "${source_dir}/${required_path}")
            message(
                FATAL_ERROR
                "${set_name} data is missing ${source_dir}/${required_path}"
            )
        endif()
    endforeach()

    if(NOT IS_DIRECTORY "${source_dir}/skfiles")
        message(
            FATAL_ERROR
            "${set_name} data is missing ${source_dir}/skfiles"
        )
    endif()

    file(GLOB slakos_files "${source_dir}/skfiles/*.skf")
    if(NOT slakos_files)
        message(FATAL_ERROR "${set_name} contains no Slater-Koster files")
    endif()
endfunction()

function(CheckoutRepository repository_url source_dir revision)
    if(NOT EXISTS "${source_dir}/.git")
        if(EXISTS "${source_dir}")
            message(
                FATAL_ERROR
                "${source_dir} exists but is not a Git checkout"
            )
        endif()

        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}" clone --no-checkout
                "${repository_url}" "${source_dir}"
            RESULT_VARIABLE clone_result
        )
        if(NOT clone_result EQUAL 0)
            message(FATAL_ERROR "Failed to clone ${repository_url}")
        endif()
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${source_dir}"
            cat-file -e "${revision}^{commit}"
        RESULT_VARIABLE revision_available
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT revision_available EQUAL 0)
        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}" -C "${source_dir}"
                fetch --depth 1 origin "${revision}"
            RESULT_VARIABLE fetch_result
        )
        if(NOT fetch_result EQUAL 0)
            message(
                FATAL_ERROR
                "Failed to fetch ${revision} from ${repository_url}"
            )
        endif()
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${source_dir}"
            checkout --detach "${revision}"
        RESULT_VARIABLE checkout_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT checkout_result EQUAL 0)
        message(FATAL_ERROR "Failed to checkout ${revision} in ${source_dir}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
        RESULT_VARIABLE revision_result
        OUTPUT_VARIABLE actual_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT revision_result EQUAL 0 OR
       NOT "${actual_revision}" STREQUAL "${revision}")
        message(
            FATAL_ERROR
            "Expected ${revision} in ${source_dir}, found ${actual_revision}"
        )
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" status --porcelain
        RESULT_VARIABLE status_result
        OUTPUT_VARIABLE checkout_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT status_result EQUAL 0 OR NOT "${checkout_status}" STREQUAL "")
        message(
            FATAL_ERROR
            "${source_dir} contains uncommitted or untracked files"
        )
    endif()
endfunction()

set(SLAKOS_3OB_REVISION "c5e165cb65f80f6b4e054c99e3f770ac3b8a4ecc")
set(SLAKOS_MATSCI_REVISION "57016b4363fd6180f6edff662e5fbaa95276b4bd")

if(PQ_SLAKOS_SOURCE_DIR)
    get_filename_component(
        SLAKOS_SOURCE_DIR
        "${PQ_SLAKOS_SOURCE_DIR}"
        ABSOLUTE
        BASE_DIR "${CMAKE_SOURCE_DIR}"
    )
    message(STATUS "Using preseeded Slater-Koster data: ${SLAKOS_SOURCE_DIR}")
else()
    find_package(Git REQUIRED)
    set(SLAKOS_SOURCE_DIR "${CMAKE_BINARY_DIR}/external/slakos")
    CheckoutRepository(
        "https://github.com/dftbparams/3ob.git"
        "${SLAKOS_SOURCE_DIR}/3ob"
        "${SLAKOS_3OB_REVISION}"
    )
    CheckoutRepository(
        "https://github.com/dftbparams/matsci.git"
        "${SLAKOS_SOURCE_DIR}/matsci"
        "${SLAKOS_MATSCI_REVISION}"
    )
endif()

ValidateSlakosSet("${SLAKOS_SOURCE_DIR}/3ob" "3ob")
ValidateSlakosSet("${SLAKOS_SOURCE_DIR}/matsci" "matsci")

# define directory for 3ob and matsci for preprocessor
add_compile_definitions(__SLAKOS_DIR__="${SLAKOS_SOURCE_DIR}/")

foreach(slakos_set IN ITEMS 3ob matsci)
    install(
        DIRECTORY "${SLAKOS_SOURCE_DIR}/${slakos_set}/skfiles"
        DESTINATION "share/PQ/slakos/${slakos_set}"
        COMPONENT slakos
    )
    install(
        FILES
            "${SLAKOS_SOURCE_DIR}/${slakos_set}/LICENSE"
            "${SLAKOS_SOURCE_DIR}/${slakos_set}/README"
            "${SLAKOS_SOURCE_DIR}/${slakos_set}/RELEASE.md"
            "${SLAKOS_SOURCE_DIR}/${slakos_set}/CHANGELOG.md"
        DESTINATION "share/PQ/slakos/${slakos_set}"
        COMPONENT slakos
    )
endforeach()
