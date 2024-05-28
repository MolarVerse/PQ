find_package(Git QUIET)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)

    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)

        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
enable_testing()
option(INSTALL_GMOCK "install Googletest's GMock?" OFF)
option(INSTALL_GTEST "install Googletest's GTest?" OFF)

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

if(${BUILD_WITH_GCOVR})
    include(gcovr)

    include(CodeCoverage)
    set(EXCLUDE_FOR_GCOVR
        "\.build"
        .build
        tests
        build/_deps # don't know why build
        apps
        external
        benchmarks
        tools
        src/setup/setup.cpp
        src/engine/*
        include/engine/*
    )

    set(GCOVR_ADDITIONAL_ARGS
        "--exclude-throw-branches"
        "--exclude-function-lines"
        "--exclude-noncode-lines"
        "--gcov-ignore-errors=no_working_dir_found"
    )

    setup_target_for_coverage_gcovr_html(
        NAME coverage
        EXECUTABLE ctest
        EXCLUDE ${EXCLUDE_FOR_GCOVR}
    )

    setup_target_for_coverage_gcovr_xml(
        NAME coverage_xml
        EXECUTABLE ctest
        EXCLUDE ${EXCLUDE_FOR_GCOVR}
    )

    setup_target_for_coverage_lcov(
        NAME coverage_lcov
        EXECUTABLE ctest
        LCOV_ARGS " --no-external "
        EXCLUDE tests* _deps* apps* external* apps* benchmarks* build* .build*
    )
endif()

add_definitions(-DWITH_TESTS)