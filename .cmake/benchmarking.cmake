# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE
)

# Keep project warning policy from being applied to third-party sources.
set(_CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
foreach(flag IN ITEMS "-Wswitch-enum" "-Werror=switch-enum")
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()

FetchContent_MakeAvailable(googlebenchmark)

set(CMAKE_CXX_FLAGS "${_CMAKE_CXX_FLAGS_BACKUP}")
