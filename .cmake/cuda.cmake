find_package(CUDA 12.4 REQUIRED)
add_definitions(-DWITH_CUDA)
include_directories(${CUDA_INCLUDE_DIRS})