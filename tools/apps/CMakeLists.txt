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
        pimd_qmcf_input
    )

    target_compile_options(${executable}
        PRIVATE
        "-D${executable}"
    )

    get_target_property(cflags ${executable} COMPILE_OPTIONS)
    message(STATUS "Compile options for ${executable}: ${cflags}")

    install(TARGETS ${executable}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tools/
    )
endforeach()