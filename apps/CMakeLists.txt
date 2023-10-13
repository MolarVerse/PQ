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
  pimd_qmcf_mpi
)

if(BUILD_WITH_MPI)
  target_link_libraries(pimd_qmcf
    PRIVATE
    MPI::MPI_C
  )
endif()

install(TARGETS pimd_qmcf
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
