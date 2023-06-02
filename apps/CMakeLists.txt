add_executable(pimd_qmcf
  pimd_qmcf.cpp
)

target_include_directories(pimd_qmcf
  PUBLIC ${CODE_INCLUDE_DIR}
)

if(BUILD_WITH_MPI)
  target_link_libraries(pimd_qmcf
    PUBLIC
    MPI::MPI_C
  )
endif()

target_link_libraries(pimd_qmcf
  PRIVATE
  PUBLIC
  setup
  simulationBox
  engine
  utilities
  config
  output
  math
)

install(TARGETS pimd_qmcf
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/bin
)
