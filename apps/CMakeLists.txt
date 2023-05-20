find_package(MPI REQUIRED)

set(executables pimd_qmcf pimd_qmcf_mpi)

foreach(executable ${executables})
  add_executable(${executable}
    ${executable}.cpp
  )

  target_include_directories(${executable}
  PUBLIC ${CODE_INCLUDE_DIR}
  )

  # target_link_libraries(${executable}
  # PUBLIC
  # MPI::MPI_C
  # )

  target_link_libraries(${executable}
    PRIVATE
    MPI::MPI_C
    PUBLIC
    setup
    simulationBox
    engine
    utilities
    config
    output
  )

  install(TARGETS ${executable}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/bin
  )
endforeach()
