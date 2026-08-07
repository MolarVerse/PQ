# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_TESTING NO)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main
)

set(googlebenchmark_have_std_regex ON CACHE INTERNAL "")
set(googlebenchmark_run_have_std_regex ON CACHE INTERNAL "")

FetchContent_MakeAvailable(googlebenchmark)