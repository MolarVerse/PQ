#ifndef __DEVICE_TPP_HPP__
#define __DEVICE_TPP_HPP__

#include "device.hpp"
#include "deviceAPI.hpp"
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

    /**
     * @brief function to copy memory from the host to the device
     *
     * @param dst the destination pointer on the device
     * @param src the source vector on the host
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyTo(T* dst, const std::vector<T>& src)
    {
        const auto error =
            deviceMemcpy(dst, src, src.size(), __deviceMemcpyHostToDevice__);

        addDeviceError(error, "Copying memory from host to device");

        return error;
    }

    /**
     * @brief function to copy memory from the host to the device
     *
     * @param dst the destination pointer on the device
     * @param src the source vector on the host
     * @param size the size of the memory to copy
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyTo(
        T*           dst,
        const T*     src,
        const size_t size
    )
    {
        const auto error =
            deviceMemcpy(dst, src, size, __deviceMemcpyHostToDevice__);

        addDeviceError(error, "Copying memory from host to device");

        return error;
    }

    /**
     * @brief function to copy memory from the device to the host
     *
     * @param dst the destination vector on the host
     * @param src the source pointer on the device
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyFrom(std::vector<T>& dst, const T* src)
    {
        const auto error = deviceMemcpy(
            dst.data(),
            src,
            dst.size(),
            __deviceMemcpyDeviceToHost__
        );

        addDeviceError(error, "Copying memory from device to host");

        return error;
    }

    /**
     * @brief function to copy memory from the device to the host
     *
     * @param dst the destination pointer on the host
     * @param src the source pointer on the device
     * @param size the size of the memory to copy
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyFrom(
        T*           dst,
        const T*     src,
        const size_t size
    )
    {
        const auto error =
            deviceMemcpy(dst, src, size, __deviceMemcpyDeviceToHost__);

        addDeviceError(error, "Copying memory from device to host");

        return error;
    }

    /**
     * @brief function to copy memory from the host to the device asynchronously
     *
     * @details this function will use per default the PQ dataStream to copy the
     * memory from the host to the device.
     *
     * @param dst the destination pointer on the device
     * @param src the source vector on the host
     * @return deviceError_t
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyToAsync(T* dst, const std::vector<T>& src)
    {
        const auto error = deviceMemcpyAsync(
            dst,
            src.data(),
            src.size(),
            __deviceMemcpyHostToDevice__,
            _dataStream
        );

        addDeviceError(
            error,
            "Asynchronous copying memory from host to device"
        );

        return error;
    }

    /**
     * @brief function to copy memory from the host to the device asynchronously
     *
     * @details this function will use per default the PQ dataStream to copy the
     * memory from the host to the device.
     *
     * @param dst the destination pointer on the device
     * @param src the source pointer on the host
     * @param size the size of the memory to copy
     * @return deviceError_t
     *
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyToAsync(
        T*           dst,
        const T*     src,
        const size_t size
    )
    {
        const auto error = deviceMemcpyAsync(
            dst,
            src,
            size,
            __deviceMemcpyHostToDevice__,
            _dataStream
        );

        addDeviceError(
            error,
            "Asynchronous copying memory from host to device"
        );

        return error;
    }

    /**
     * @brief function to copy memory from the device to the host asynchronously
     *
     * @details this function will use per default the PQ dataStream to copy the
     *
     * @param dst the destination vector on the host
     * @param src the source pointer on the device
     * @return deviceError_t
     *
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyFromAsync(
        std::vector<T>& dst,
        const T*        src
    )
    {
        const auto error = deviceMemcpyAsync(
            dst.data(),
            src,
            dst.size(),
            __deviceMemcpyDeviceToHost__,
            _dataStream
        );

        addDeviceError(
            error,
            "Asynchronous copying memory from device to host"
        );

        return error;
    }

    /**
     * @brief function to copy memory from the device to the host asynchronously
     *
     * @details this function will use per default the PQ dataStream to copy the
     *
     * @param dst the destination pointer on the host
     * @param src the source pointer on the device
     * @param size the size of the memory to copy
     * @return deviceError_t
     *
     */
    template <typename T>
    deviceError_t Device::deviceMemcpyFromAsync(
        T*           dst,
        const T*     src,
        const size_t size
    )
    {
        const auto error = deviceMemcpyAsync(
            dst,
            src,
            size,
            __deviceMemcpyDeviceToHost__,
            _dataStream
        );

        addDeviceError(
            error,
            "Asynchronous copying memory from device to host"
        );

        return error;
    }

}   // namespace device

#endif   // __DEVICE_TPP_HPP__
