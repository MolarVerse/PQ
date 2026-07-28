add_executable(PQ
  PQ.cpp
  capabilities.cpp
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
  PQ_BUILD_WITH_KOKKOS=$<BOOL:${BUILD_WITH_KOKKOS}>
  PQ_BUILD_WITH_PYTHON_BINDINGS=$<BOOL:${BUILD_WITH_PYTHON_BINDINGS}>
  PQ_BUILD_WITH_PYTHON_EMBEDDING=$<BOOL:${BUILD_WITH_PYBIND11}>
  PQ_BUILD_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>
  PQ_BUILD_STATIC=$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>
  PQ_BUILD_WITH_SINGULARITY=$<BOOL:${BUILD_WITH_SINGULARITY}>
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
    NAME testPQCli
    COMMAND
      ${CMAKE_COMMAND}
      -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
      -DEXPECTED_VERSION=${GIT_VERSION}
      -DEXPECTED_ASE=${BUILD_WITH_ASE}
      -DEXPECTED_MPI=${BUILD_WITH_MPI}
      -DEXPECTED_KOKKOS=${BUILD_WITH_KOKKOS}
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
      -DEXPECTED_ASE=${BUILD_WITH_ASE}
      -DEXPECTED_SHARED=${BUILD_SHARED_LIBS}
      -DEXPECTED_SINGULARITY=${BUILD_WITH_SINGULARITY}
      -DVALIDATION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/regression/fixtures/mm-md-smoke
      -DVALIDATION_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/pq-cli-validation
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQValidation.cmake
  )
  set_property(TEST testPQValidation PROPERTY LABELS input)

  if(BUILD_WITH_ASE)
    add_test(
      NAME testPQSlakosInstall
      COMMAND
        ${CMAKE_COMMAND}
        -DPQ_EXECUTABLE=$<TARGET_FILE:PQ>
        -DBUILD_DIR=${CMAKE_BINARY_DIR}
        -DSLAKOS_SOURCE_DIR=${SLAKOS_SOURCE_DIR}
        -DSTAGING_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pq-slakos-install
        -DMM_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/regression/fixtures/mm-md-smoke
        -P ${PROJECT_SOURCE_DIR}/tests/cmake/testPQSlakosInstall.cmake
    )
    set_property(TEST testPQSlakosInstall PROPERTY LABELS input)
  endif()
endif()
