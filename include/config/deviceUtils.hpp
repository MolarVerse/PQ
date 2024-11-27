#ifndef __DEVICE_UTILS_HPP__
#define __DEVICE_UTILS_HPP__

// NOTE: far in the future maybe HIP support

// clang-format off

#ifdef __PQ_GPU__

/*********************************
 *                               *
 * Clean up the macro namespaces *
 *                               *
 *********************************/

#if defined(__HIPCC__) && !defined(__PQ_CUDA__)
    #define __PQ_HIP__
#elif defined(__CUDACC__)
    #define __PQ_CUDA__
#endif

/*******************************
 *                             *
 * Include the correct headers *
 *                             *
 *******************************/

#if defined(__PQ_HIP__)
    #include <hip/hip_runtime.h>
#elif defined(__PQ_CUDA__)
    #include <cuda_runtime.h>
#endif

/********************************
 *                              *
 * Device memory allocation API *
 *                              *
 ********************************/

/**
 * @brief Wrapper function for device memory allocation
 * 
 * @param ptr 
 * @param size 
 * @return auto 
 */
auto inline deviceMalloc(void **ptr, size_t size)
{
#if defined(__PQ_HIP__)
    return hipMalloc(ptr, size);
#elif defined(__PQ_CUDA__)
    return cudaMalloc(ptr, size);
#endif
}

#endif  // __PQ_GPU__

// clang-format on

#endif   // __DEVICE_UTILS_HPP__