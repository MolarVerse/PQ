#ifndef __DEVICE_TPP_HPP__
#define __DEVICE_TPP_HPP__

#include "device.hpp"
#include "deviceConfig.hpp"

namespace device
{
    /**
     * @brief Wrapper function for device memory allocation
     *
     * @param ptr
     * @param size
     * @return deviceError_t
     */
    template <typename T>
    void Device::deviceMalloc(T** ptr, size_t size)
    {
        const auto error =
            __DEVICE_MALLOC(reinterpret_cast<void**>(ptr), size * sizeof(T));

        if (error != deviceSuccess)
            _errorMsgs.push_back(
                "Device memory allocation failed with the following error: " +
                deviceGetErrorString(error)
            );
    }
}   // namespace device

#endif   // __DEVICE_TPP_HPP__
