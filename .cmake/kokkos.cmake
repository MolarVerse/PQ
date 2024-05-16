include(FetchContent)
FetchContent_Declare(kokkos
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(kokkos-kernels
    GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
)
FetchContent_MakeAvailable(kokkos-kernels)

add_definitions(-DWITH_KOKKOS)
