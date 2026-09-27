add_library(pq_test_main
    main.cpp
)

target_link_libraries(pq_test_main
    PUBLIC
    gtest
    gmock
)

if(BUILD_WITH_MPI)
    target_link_libraries(pq_test_main
        PUBLIC
        PQ_mpi
        MPI::MPI_C
    )
endif()
