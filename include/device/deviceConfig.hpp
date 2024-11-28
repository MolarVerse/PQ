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

#ifndef __DEVICE_UTILS_HPP__
#define __DEVICE_UTILS_HPP__

#ifdef __PQ_GPU__

// clang-format off

/*********************************
 *                               *
 * Clean up the macro namespaces *
 *                               *
 *********************************/

#if defined(__HIPCC__)
    #define __PQ_HIP__
#elif defined(__PQ_NVCXX__)
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

/*******************************************
 *                                         *
 * Define the initialization device macros *
 *                                         *
 *******************************************/

#if defined(__PQ_HIP__)
    #define __getDevice(x) hipGetDevice(x)
#elif defined(__PQ_CUDA__)
    #define __getDevice(x) cudaGetDevice(x)
#endif

#if defined(__PQ_HIP__)
    #define __getDeviceCount(x) hipGetDeviceCount(x)
#elif defined(__PQ_CUDA__)
    #define __getDeviceCount(x) cudaGetDeviceCount(x)
#endif

/************************************
 *                                  *
 * Define the __DEVICE_MALLOC macro *
 *                                  *
 ************************************/

#if defined(__PQ_HIP__)
    #define __deviceMalloc(ptr, size) hipMalloc(ptr, size)
#elif defined(__PQ_CUDA__)
    #define __deviceMalloc(ptr, size) cudaMalloc(ptr, size)
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

#if defined(__PQ_HIP__)
    using deviceError_t = hipError_t;
#elif defined(__PQ_CUDA__)
    using deviceError_t = cudaError_t;
#endif

#endif   // __PQ_GPU__

#endif   // __DEVICE_UTILS_HPP__