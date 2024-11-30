#ifndef __DEVICE_API_HPP__
#define __DEVICE_API_HPP__

#include <cstddef>   // for size_t

#include "deviceConfig.hpp"

namespace device
{

    template <typename T>
    deviceError_t deviceMemcpy(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind
    );

    template <typename T>
    deviceError_t deviceMemcpyAsync(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind,
        const deviceStream_t     stream
    );

}   // namespace device

#include "deviceAPI.tpp.hpp"   // DO NOT MOVE THIS LINE!!!

#endif __DEVICE_API_HPP__
