#ifndef __DEVICE_API_TPP__
#define __DEVICE_API_TPP__

#include "deviceAPI.hpp"
#include "deviceConfig.hpp"

namespace device
{
    /**
     * @brief This function is an API wrapper for the device Memcpy function
     *
     * @tparam T
     * @param dst
     * @param src
     * @param size
     * @param kind
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t deviceMemcpy(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind
    )
    {
        return __deviceMemcpy(
            reinterpret_cast<void*>(dst),
            reinterpret_cast<const void*>(src),
            size * sizeof(T),
            kind
        );
    }

    /**
     * @brief This function is an API wrapper for the device MemcpyAsync
     * function
     *
     * @tparam T
     * @param dst
     * @param src
     * @param size
     * @param kind
     * @param stream
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t deviceMemcpyAsync(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind,
        const deviceStream_t     stream
    )
    {
        return __deviceMemcpyAsync(
            reinterpret_cast<void*>(dst),
            reinterpret_cast<const void*>(src),
            size * sizeof(T),
            kind,
            stream
        );
    }
}   // namespace device

#endif   // __DEVICE_API_TPP__