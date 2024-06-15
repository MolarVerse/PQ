find_package(CUDA 12.4 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DWITH_CUDA)