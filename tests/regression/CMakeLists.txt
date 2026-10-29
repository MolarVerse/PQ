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
