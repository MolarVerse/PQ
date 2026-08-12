add_executable(PQ
  PQ.cpp
  capabilities.cpp
  jsonOutput.cpp
  validation.cpp
)

target_include_directories(PQ
  PUBLIC ${CODE_INCLUDE_DIR}
)

if(NOT DEFINED BUILD_WITH_PYBIND11)
  set(BUILD_WITH_PYBIND11 OFF)
endif()

target_compile_definitions(PQ
  PRIVATE
  PQ_BUILD_WITH_ASE=$<BOOL:${BUILD_WITH_ASE}>
  PQ_BUILD_WITH_MPI=$<BOOL:${BUILD_WITH_MPI}>
  PQ_BUILD_WITH_PYTHON_BINDINGS=$<BOOL:${BUILD_WITH_PYTHON_BINDINGS}>
  PQ_BUILD_WITH_PYTHON_EMBEDDING=$<BOOL:${BUILD_WITH_PYBIND11}>
  PQ_BUILD_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>
  PQ_BUILD_STATIC=$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>
  PQ_BUILD_WITH_SINGULARITY=$<BOOL:${BUILD_WITH_SINGULARITY}>
  PQ_BUILD_EXECUTABLE_DIR="${PROJECT_BINARY_DIR}/apps"
  PQ_BUILD_QM_SCRIPT_DIR="${PROJECT_BINARY_DIR}/src/QM/scripts"
  PQ_BUILD_SLAKOS_DIR="${PROJECT_BINARY_DIR}/external/slakos"
)

target_link_libraries(PQ
  PUBLIC
  PQ_input
  setup
  engine
)

if(BUILD_WITH_MPI)
  target_link_libraries(PQ
    PRIVATE
    MPI::MPI_C
    PQ_mpi
  )
endif()

if(BUILD_WITH_PYBIND11)
  target_link_libraries(PQ
    PRIVATE
    pybind11::embed
    pybind11::module
  )
endif()

install(TARGETS PQ
  DESTINATION bin
)

if(BUILD_WITH_TESTS)
  add_test(
    NAME testPQReferenceInstall
    COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_DIR=${CMAKE_BINARY_DIR}
      -DREFERENCE_TEST_EXECUTABLE=$<TARGET_FILE:testReferencesOutput>
      -DSTAGING_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pq-reference-install
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQReferenceInstall.cmake
  )
  set_tests_properties(
    testPQReferenceInstall
    PROPERTIES LABELS installation RUN_SERIAL TRUE
  )

  add_test(
    NAME testPQExternalQMScriptInstall
    COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_DIR=${CMAKE_BINARY_DIR}
      -DSCRIPT_TEST_EXECUTABLE=$<TARGET_FILE:testQMSetup>
      -DSTAGING_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pq-qm-script-install
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQExternalQMScriptInstall.cmake
  )
  set_tests_properties(
    testPQExternalQMScriptInstall
    PROPERTIES LABELS installation RUN_SERIAL TRUE
  )

  if(BUILD_WITH_ASE)
    add_test(
      NAME testPQSlakosInstall
      COMMAND
        ${CMAKE_COMMAND}
        -DBUILD_DIR=${CMAKE_BINARY_DIR}
        -DSLAKOS_SOURCE_DIR=${SLAKOS_SOURCE_DIR}
        -DSLAKOS_TEST_EXECUTABLE=$<TARGET_FILE:testQMSettings>
        -DSTAGING_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pq-slakos-install
        -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQSlakosInstall.cmake
    )
    set_tests_properties(
      testPQSlakosInstall
      PROPERTIES LABELS installation RUN_SERIAL TRUE
    )
  endif()

  add_test(
    NAME testPQCli
    COMMAND
      ${CMAKE_COMMAND}
      -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
      -DEXPECTED_VERSION=${GIT_VERSION}
      -DEXPECTED_ASE=${BUILD_WITH_ASE}
      -DEXPECTED_MPI=${BUILD_WITH_MPI}
      -DEXPECTED_PYTHON_BINDINGS=${BUILD_WITH_PYTHON_BINDINGS}
      -DEXPECTED_PYTHON_EMBEDDING=${BUILD_WITH_PYBIND11}
      -DEXPECTED_SHARED=${BUILD_SHARED_LIBS}
      -DEXPECTED_SINGULARITY=${BUILD_WITH_SINGULARITY}
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQCli.cmake
  )
  set_property(TEST testPQCli PROPERTY LABELS input)

  add_test(
    NAME testPQValidation
    COMMAND
      ${CMAKE_COMMAND}
      -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
      -DVALIDATION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/regression/fixtures/mm-md-smoke
      -DVALIDATION_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/pq-cli-validation
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQValidation.cmake
  )
  set_property(TEST testPQValidation PROPERTY LABELS input)

  add_test(
    NAME testPQValidationWarnings
    COMMAND
      ${CMAKE_COMMAND}
      -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
      -DVALIDATION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/regression/fixtures/mm-md-smoke
      -DVALIDATION_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/pq-validation-warnings
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQValidationWarnings.cmake
  )
  set_property(TEST testPQValidationWarnings PROPERTY LABELS input)

  add_test(
    NAME testPQValidationEnvironment
    COMMAND
      ${CMAKE_COMMAND}
      -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
      -DEXPECTED_ASE=${BUILD_WITH_ASE}
      -DEXPECTED_SHARED=${BUILD_SHARED_LIBS}
      -DEXPECTED_SINGULARITY=${BUILD_WITH_SINGULARITY}
      -DBUNDLED_SCRIPT_SOURCE_DIR=${PROJECT_SOURCE_DIR}/src/QM/scripts
      -DVALIDATION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/regression/fixtures/mm-md-smoke
      -DVALIDATION_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/pq-validation-environment
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQValidationEnvironment.cmake
  )
  set_property(TEST testPQValidationEnvironment PROPERTY LABELS input)
endif()
