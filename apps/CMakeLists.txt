add_executable(pimd_qmcf
  pimd_qmcf.cpp
)

target_include_directories(pimd_qmcf
  PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(pimd_qmcf
  PUBLIC
  pimd_qmcf_input
  setup
  engine
)

if(BUILD_WITH_MPI)
  target_link_libraries(pimd_qmcf
    PRIVATE
    MPI::MPI_C
    pimd_qmcf_mpi
  )
endif()

install(TARGETS pimd_qmcf
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
