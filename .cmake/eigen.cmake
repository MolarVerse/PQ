message(CHECK_START "Fetching Eigen3")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(EIGEN_TAG 5.0.0)

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG ${EIGEN_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen)
FetchContent_GetProperties(Eigen)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
