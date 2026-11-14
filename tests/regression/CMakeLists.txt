find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(REGRESSION_CASE mm-md-smoke)
set(REGRESSION_CASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${REGRESSION_CASE})
set(REGRESSION_CASE_ZIP ${CMAKE_CURRENT_BINARY_DIR}/${REGRESSION_CASE}.zip)

set(REGRESSION_CASE_FILES
    run.in
    start.rst
    moldescriptor.dat
    guff.dat
    manifest.json
)

set(REGRESSION_CASE_FILE_PATHS)
foreach(REGRESSION_CASE_FILE ${REGRESSION_CASE_FILES})
    list(APPEND
        REGRESSION_CASE_FILE_PATHS
        ${REGRESSION_CASE_DIR}/${REGRESSION_CASE_FILE}
    )
endforeach()

add_custom_command(
    OUTPUT ${REGRESSION_CASE_ZIP}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${REGRESSION_CASE_ZIP}
    COMMAND
        ${CMAKE_COMMAND} -E tar cf ${REGRESSION_CASE_ZIP}
        --format=zip -- ${REGRESSION_CASE_FILES}
    WORKING_DIRECTORY ${REGRESSION_CASE_DIR}
    DEPENDS ${REGRESSION_CASE_FILE_PATHS}
    VERBATIM
)

add_custom_target(regression_example_zips ALL DEPENDS ${REGRESSION_CASE_ZIP})

add_test(
    NAME regression_mm_md_smoke_zip
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_regression_case.py
        --pq $<TARGET_FILE:PQ>
        --case-zip ${REGRESSION_CASE_ZIP}
        --manifest manifest.json
)

set_property(TEST regression_mm_md_smoke_zip PROPERTY LABELS regression)
set_property(TEST regression_mm_md_smoke_zip PROPERTY TIMEOUT 30)

set(MSHAKE_REGRESSION_CASE mm-md-shake-mshake)
set(
    MSHAKE_REGRESSION_CASE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${MSHAKE_REGRESSION_CASE}
)
set(
    MSHAKE_REGRESSION_CASE_ZIP
    ${CMAKE_CURRENT_BINARY_DIR}/${MSHAKE_REGRESSION_CASE}.zip
)

set(MSHAKE_REGRESSION_CASE_FILES
    shake.in
    mshake.in
    start.rst
    shake.top
    mshake.top
    empty.top
    moldescriptor.dat
    guff.dat
    manifest.json
)

set(MSHAKE_REGRESSION_CASE_FILE_PATHS)
foreach(MSHAKE_REGRESSION_CASE_FILE ${MSHAKE_REGRESSION_CASE_FILES})
    list(APPEND
        MSHAKE_REGRESSION_CASE_FILE_PATHS
        ${MSHAKE_REGRESSION_CASE_DIR}/${MSHAKE_REGRESSION_CASE_FILE}
    )
endforeach()

add_custom_command(
    OUTPUT ${MSHAKE_REGRESSION_CASE_ZIP}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${MSHAKE_REGRESSION_CASE_ZIP}
    COMMAND
        ${CMAKE_COMMAND} -E tar cf ${MSHAKE_REGRESSION_CASE_ZIP}
        --format=zip -- ${MSHAKE_REGRESSION_CASE_FILES}
    WORKING_DIRECTORY ${MSHAKE_REGRESSION_CASE_DIR}
    DEPENDS ${MSHAKE_REGRESSION_CASE_FILE_PATHS}
    VERBATIM
)

add_custom_target(
    regression_mshake_example_zip ALL
    DEPENDS ${MSHAKE_REGRESSION_CASE_ZIP}
)

add_test(
    NAME regression_shake_mshake_equivalence_zip
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_regression_case.py
        --pq $<TARGET_FILE:PQ>
        --case-zip ${MSHAKE_REGRESSION_CASE_ZIP}
        --manifest manifest.json
)

set_property(
    TEST regression_shake_mshake_equivalence_zip
    PROPERTY LABELS regression
)
set_property(TEST regression_shake_mshake_equivalence_zip PROPERTY TIMEOUT 60)

if(BUILD_WITH_MPI)
    set(MPI_REGRESSION_CASE qm-rpmd-mpi-smoke)
    set(
        MPI_REGRESSION_CASE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${MPI_REGRESSION_CASE}
    )
    set(
        MPI_REGRESSION_CASE_ZIP
        ${CMAKE_CURRENT_BINARY_DIR}/${MPI_REGRESSION_CASE}.zip
    )

    set(MPI_REGRESSION_CASE_FILES
        run.in
        start.rst
        fake_qm.py
        manifest.json
    )

    set(MPI_REGRESSION_CASE_FILE_PATHS)
    foreach(MPI_REGRESSION_CASE_FILE ${MPI_REGRESSION_CASE_FILES})
        list(APPEND
            MPI_REGRESSION_CASE_FILE_PATHS
            ${MPI_REGRESSION_CASE_DIR}/${MPI_REGRESSION_CASE_FILE}
        )
    endforeach()

    add_custom_command(
        OUTPUT ${MPI_REGRESSION_CASE_ZIP}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${MPI_REGRESSION_CASE_ZIP}
        COMMAND
            ${CMAKE_COMMAND} -E tar cf ${MPI_REGRESSION_CASE_ZIP}
            --format=zip -- ${MPI_REGRESSION_CASE_FILES}
        WORKING_DIRECTORY ${MPI_REGRESSION_CASE_DIR}
        DEPENDS ${MPI_REGRESSION_CASE_FILE_PATHS}
        VERBATIM
    )

    add_custom_target(
        regression_mpi_example_zip ALL
        DEPENDS ${MPI_REGRESSION_CASE_ZIP}
    )

    add_test(
        NAME regression_mpi_qm_rpmd_smoke_zip
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_regression_case.py
            --pq $<TARGET_FILE:PQ>
            --case-zip ${MPI_REGRESSION_CASE_ZIP}
            --manifest manifest.json
            --launcher
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG}
            2
    )

    set_property(
        TEST regression_mpi_qm_rpmd_smoke_zip
        PROPERTY LABELS regression
    )
    set_property(TEST regression_mpi_qm_rpmd_smoke_zip PROPERTY TIMEOUT 30)
endif()

# ORCA: real PQ binary and bundled ORCA drivers, fake ORCA executable.
function(pq_add_orca_regression NAME MANIFEST TIMEOUT)
    set(stage ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    set(zip ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.zip)
    set(names)
    foreach(file ${ARGN})
        get_filename_component(name ${file} NAME)
        list(APPEND names ${name})
    endforeach()
    add_custom_command(
        OUTPUT ${zip}
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${stage} ${zip}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${stage}
        COMMAND ${CMAKE_COMMAND} -E copy ${ARGN} ${stage}
        COMMAND
            ${CMAKE_COMMAND} -E chdir ${stage}
            ${CMAKE_COMMAND} -E tar cf ${zip} --format=zip -- ${names}
        DEPENDS ${ARGN}
        VERBATIM
    )
    add_custom_target(${NAME}_zip ALL DEPENDS ${zip})
    add_test(
        NAME ${NAME}
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_regression_case.py
            --pq $<TARGET_FILE:PQ>
            --case-zip ${zip}
            --manifest ${MANIFEST}
    )
    set_property(TEST ${NAME} PROPERTY LABELS "regression;QM")
    set_property(TEST ${NAME} PROPERTY TIMEOUT ${TIMEOUT})
endfunction()

set(ORCA_FIXTURE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/orca-fake)
set(ORCA_EXAMPLE ${PROJECT_SOURCE_DIR}/examples/orca)
set(ORCA_COMMON
    ${ORCA_FIXTURE}/fake_orca.py
    ${ORCA_FIXTURE}/check_orca.py
    ${ORCA_EXAMPLE}/orca.template
    ${ORCA_EXAMPLE}/orca_hf.template
    ${ORCA_EXAMPLE}/orca_dft.template
)

pq_add_orca_regression(regression_orca_qm_fake manifest-qm.json 60
    ${ORCA_COMMON}
    ${ORCA_FIXTURE}/manifest-qm.json
    ${ORCA_FIXTURE}/qm-hf.in
    ${ORCA_FIXTURE}/qm-dft.in
    ${ORCA_FIXTURE}/qm-rimp2.in
    ${ORCA_EXAMPLE}/water.rst
)

set(QMMM_EXAMPLE ${PROJECT_SOURCE_DIR}/examples/potassium_water_qmmm)
pq_add_orca_regression(regression_orca_qmmm_fake manifest-qmmm.json 120
    ${ORCA_COMMON}
    ${ORCA_FIXTURE}/manifest-qmmm.json
    ${ORCA_FIXTURE}/qmmm-dft.in
    ${ORCA_FIXTURE}/qmmm-rimp2.in
    ${QMMM_EXAMPLE}/input.rst
    ${QMMM_EXAMPLE}/moldescriptor.dat
    ${QMMM_EXAMPLE}/guff.dat
)

# Real ORCA, only where it is installed: -DORCA_EXECUTABLE=/path/to/orca.
# Only the quantum chemistry ORCA banner counts (not the GNOME screen reader).
find_program(ORCA_EXECUTABLE orca)
set(PQ_ORCA_VALID FALSE)
if(ORCA_EXECUTABLE)
    execute_process(
        COMMAND ${ORCA_EXECUTABLE} --version
        TIMEOUT 10
        OUTPUT_VARIABLE orca_output
        ERROR_QUIET
    )
    if(orca_output MATCHES "O +R +C +A")
        set(PQ_ORCA_VALID TRUE)
    else()
        message(STATUS
            "${ORCA_EXECUTABLE} is not quantum chemistry ORCA: skip regression_orca_real")
    endif()
endif()
if(PQ_ORCA_VALID)
    add_test(
        NAME regression_orca_real
        COMMAND
            ${CMAKE_COMMAND} -E env ORCA_EXE=${ORCA_EXECUTABLE}
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/orca-real/validate_orca.py
            ${PROJECT_SOURCE_DIR}/src/QM/scripts
            ${ORCA_EXAMPLE}
            ${CMAKE_CURRENT_BINARY_DIR}/orca-real
    )
    set_property(TEST regression_orca_real PROPERTY LABELS "regression;QM;orca")
    set_property(TEST regression_orca_real PROPERTY TIMEOUT 1800)
endif()
