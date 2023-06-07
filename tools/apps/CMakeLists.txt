set(EXECUTABLES analysis trajectoryToCenterOfMass)

foreach(executable ${EXECUTABLES})
    add_executable(${executable}
        analysis.cpp
    )

    target_include_directories(${executable}
        PUBLIC ${CODE_INCLUDE_DIR}
    )

    if(BUILD_WITH_MPI)
        target_link_libraries(${executable}
            PUBLIC
            MPI::MPI_C
        )
    endif()

    target_link_libraries(${executable}
        PUBLIC
        engineTools
        config
    )

    install(TARGETS ${executable}
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/tools/
    )
endforeach()