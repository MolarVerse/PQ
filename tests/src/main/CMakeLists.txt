add_library(pq_test_main
    main.cpp
)

target_link_libraries(pq_test_main
    PUBLIC
    gtest
    gmock
)

if(BUILD_WITH_KOKKOS)
    target_link_libraries(pq_test_main
        PUBLIC
        Kokkos::kokkos
    )

    target_include_directories(pq_test_main
        PUBLIC
        ${Kokkos_INCLUDE_DIRS_RET}
    )
endif()

install(TARGETS pq_test_main
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
