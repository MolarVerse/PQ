#ifndef __DEVICE_API_TPP__
#define __DEVICE_API_TPP__

#include <format>
#include <string>

#include "deviceAPI.hpp"
#include "deviceConfig.hpp"

namespace device
{
    /**
     * @brief This function is an API wrapper for the device Free function
     *
     * @details This function is used to deallocate memory on the device. The
     * function is a wrapper for the device API call to deallocate memory on the
     * device. The function will throw an exception if the device API call
     * fails. The exception will contain the error message and the message given
     * by the user. The function will return the error code from the device API
     * call.
     *
     * @tparam T
     * @param ptr
     * @param msg
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t deviceFreeThrowError(T* ptr, const std::string& msg)
    {
        const auto error = deviceFree(ptr);

        if (error != __deviceSuccess__)
        {
            throw DeviceException(std::format(
                "Error occurred during the device memory deallocation in: "
                "{}\n{}",
                msg,
                deviceGetErrorString(error)
            ));
        }

        return error;
    }

    /**
     * @brief This function is an API wrapper for the device Free function
     *
     * @tparam T
     * @param ptr
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t deviceFree(T* ptr)
    {
        return __deviceFree(reinterpret_cast<void*>(ptr));
    }

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