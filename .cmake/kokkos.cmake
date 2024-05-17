include(FetchContent)

find_package(Kokkos QUIET)

if(NOT Kokkos_FOUND)
    FetchContent_Declare(kokkos
        GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    )
    FetchContent_MakeAvailable(kokkos)
endif()

find_package(KokkosKernels QUIET)

if(NOT KokkosKernels_FOUND)
    FetchContent_Declare(kokkos-kernels
        GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
    )
    FetchContent_MakeAvailable(kokkos-kernels)
endif()

set(KOKKOS_CXX_STANDARD 20)

add_definitions(-DWITH_KOKKOS)
