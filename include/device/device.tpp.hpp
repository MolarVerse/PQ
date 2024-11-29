#ifndef __DEVICE_TPP_HPP__
#define __DEVICE_TPP_HPP__

#include "device.hpp"
#include "deviceConfig.hpp"

namespace device
{
    /**
     * @brief Wrapper function for device memory allocation
     *
     * @note there is a huge difference between the deviceMalloc and the
     * common hipMalloc or cudaMalloc API. While the hipMalloc and cudaMalloc
     * functions need the size in bytes, the deviceMalloc function needs the
     * size in elements. This is because the deviceMalloc function will
     * automatically multiply the size with the size of the type T.
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
    deviceError_t Device::deviceMalloc(T** ptr, const size_t size)
    {
        const auto error =
            __deviceMalloc(reinterpret_cast<void**>(ptr), size * sizeof(T));

        addDeviceError(error, "Device memory allocation");

        return error;
    }

    /**
     * @brief Wrapper function for device memory deallocation
     *
     * @details This function is used to deallocate memory on the device. The
     * function is a wrapper for the device API call to deallocate memory on the
     * device. The function will not throw an exception if the device API call
     * fails. Instead, the function will store the error message in the error
     * message list.
     *
     * @param ptr
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceFree(T* ptr)
    {
        const auto error = __deviceFree(reinterpret_cast<void*>(ptr));

        addDeviceError(error, "Device memory deallocation");

        return error;
    }
}   // namespace device

#endif   // __DEVICE_TPP_HPP__
