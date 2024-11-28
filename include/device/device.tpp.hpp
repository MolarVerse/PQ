#ifndef __DEVICE_TPP_HPP__
#define __DEVICE_TPP_HPP__

#include "device.hpp"
#include "deviceConfig.hpp"

namespace device
{
    /**
     * @brief Wrapper function for device memory allocation
     *
     * @details This function is used to allocate memory on the device. The
     * function is a wrapper for the device API call to allocate memory on the
     * device. The function will not throw an exception if the device API call
     * fails. Instead, the function will store the error message in the error
     * message list.
     *
     * @param ptr
     * @param size
     * @return deviceError_t
     */
    template <typename T>
    void Device::deviceMalloc(T** ptr, size_t size)
    {
        const auto error =
            __deviceMalloc(reinterpret_cast<void**>(ptr), size * sizeof(T));

        addDeviceError(error, "Device memory allocation");
    }
}   // namespace device

#endif   // __DEVICE_TPP_HPP__
