/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef __DEVICE_CONFIG_HPP__
#define __DEVICE_CONFIG_HPP__

#ifdef __PQ_GPU__

// clang-format off

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

/***************************
 *                         *
 * Define the device types *
 *                         *
 ***************************/

#if defined(__PQ_HIP__)
    using deviceProp_t = hipDeviceProp_t;
#elif defined(__PQ_CUDA__)
    using deviceProp_t = cudaDeviceProp;
#endif

#if defined(__PQ_HIP__)
    using deviceError_t = hipError_t;
#elif defined(__PQ_CUDA__)
    using deviceError_t = cudaError_t;
#endif

#if defined(__PQ_HIP__)
    using deviceStream_t = hipStream_t;
#elif defined(__PQ_CUDA__)
    using deviceStream_t = cudaStream_t;
#endif

#if defined(__PQ_HIP__)
    using deviceMemcpyKind_t = hipMemcpyKind;
#elif defined(__PQ_CUDA__)
    using deviceMemcpyKind_t = cudaMemcpyKind;
#endif

/*******************************************
 *                                         *
 * Define the initialization device macros *
 *                                         *
 *******************************************/

#if defined(__PQ_HIP__)
    #define __getDevice(x)      hipGetDevice(x)
    #define __setDevice(x)      hipSetDevice(x)
    #define __getDeviceCount(x) hipGetDeviceCount(x)
#elif defined(__PQ_CUDA__)
    #define __getDevice(x)      cudaGetDevice(x)
    #define __setDevice(x)      cudaSetDevice(x)
    #define __getDeviceCount(x) cudaGetDeviceCount(x)
#endif

#if defined(__PQ_HIP__)
    #define __getDeviceProperties(x, y) hipGetDeviceProperties((x), (y))
#elif defined(__PQ_CUDA__)
    #define __getDeviceProperties(x, y) cudaGetDeviceProperties((x), (y))
#endif

#if defined(__PQ_HIP__)
    #define __deviceStreamCreate(x)  hipStreamCreate(x)
    #define __deviceStreamDestroy(x) hipStreamDestroy(x)
#elif defined(__PQ_CUDA__)
    #define __deviceStreamCreate(x)  cudaStreamCreate(x)
    #define __deviceStreamDestroy(x) cudaStreamDestroy(x)
#endif


/************************************
 *                                  *
 * Define the __DEVICE_MALLOC macro *
 *                                  *
 ************************************/

#if defined(__PQ_HIP__)
    #define __deviceMalloc(ptr, size) hipMalloc((ptr), (size))
    #define __deviceFree(ptr)         hipFree(ptr)
#elif defined(__PQ_CUDA__)
    #define __deviceMalloc(ptr, size) cudaMalloc((ptr), (size))
    #define __deviceFree(ptr)         cudaFree(ptr)
#endif

/************************************
 *                                  *
 * Define the device syncing macros *
 *                                  *
 ************************************/

#if defined(__PQ_HIP__)
    #define __deviceSynchronize() hipDeviceSynchronize()
#elif defined(__PQ_CUDA__)
    #define __deviceSynchronize() cudaDeviceSynchronize()
#endif

#if defined(__PQ_HIP__)
    #define __deviceStreamSynchronize(x) hipStreamSynchronize(x)
#elif defined(__PQ_CUDA__)
    #define __deviceStreamSynchronize(x) cudaStreamSynchronize(x)
#endif

#if defined(__PQ_HIP__)
    #define __deviceMemcpyHostToDevice__ hipMemcpyHostToDevice
    #define __deviceMemcpyDeviceToHost__ hipMemcpyDeviceToHost
#elif defined(__PQ_CUDA__)
    #define __deviceMemcpyHostToDevice__ cudaMemcpyHostToDevice
    #define __deviceMemcpyDeviceToHost__ cudaMemcpyDeviceToHost
#endif

#if defined(__PQ_HIP__)
    #define __deviceMemcpy(dst, src, size, kind)              hipMemcpy((dst), (src), (size), (kind))
    #define __deviceMemcpyAsync(dst, src, size, kind, stream) hipMemcpyAsync((dst), (src), (size), (kind), (stream))
#elif defined(__PQ_CUDA__)
    #define __deviceMemcpy(dst, src, size, kind)              cudaMemcpy((dst), (src), (size), (kind))
    #define __deviceMemcpyAsync(dst, src, size, kind, stream) cudaMemcpyAsync((dst), (src), (size), (kind), (stream))
#endif

/*********************************
 *                               *
 * Error handling for device API *
 *                               *
 *********************************/

#if defined(__PQ_HIP__)
    #define __deviceSuccess__ hipSuccess
#elif defined(__PQ_CUDA__)
    #define __deviceSuccess__ cudaSuccess
#endif

#if defined(__PQ_HIP__)
    #define __deviceGetErrorString(x) hipGetErrorString(x)
#elif defined(__PQ_CUDA__)
    #define __deviceGetErrorString(x) cudaGetErrorString(x)
#endif

#endif   // __PQ_GPU__

#endif   // __DEVICE_CONFIG_HPP__