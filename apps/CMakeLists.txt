add_executable(PQ
  PQ.cpp
)

target_include_directories(PQ
  PUBLIC ${CODE_INCLUDE_DIR}
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
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
