enable_testing()
find_package(GTest CONFIG) # use googletest

if(NOT ${GTest_FOUND})
    message("GTest not found. Fetching...")
    include(FetchContent)
    FetchContent_Declare(
        gtest
        GIT_REPOSITORY "https://github.com/google/googletest"
        GIT_TAG main
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(gtest)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    include(gcov)

    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage EXECUTABLE ctest EXCLUDE tests.*)
endif()

add_subdirectory(tests)