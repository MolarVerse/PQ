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

#ifndef __DEVICE_HPP__
#define __DEVICE_HPP__

#include <cstddef>   // for size_t
#include <exception>
#include <string>
#include <vector>

#include "deviceConfig.hpp"

namespace device
{
    /**
     * @class Device
     *
     * @brief Contains all the information needed to run the simulation on the
     * device
     *
     * @TODO:
     *      - generalize this whole class to use multiple devices
     *      - add memcpy
     *      - add memcpyAsync
     *      - add streams with different priorities
     */
    class Device
    {
       private:
        bool _useDevice;

        int _deviceID;
        int _deviceCount;

        deviceProp_t   _deviceProp;
        deviceStream_t _dataStream;
        deviceStream_t _computeStream;
        int            _uncaughtExceptions = std::uncaught_exceptions();

        std::vector<std::string> _errorMsgs;

        void addDeviceError(const deviceError_t error, const std::string& msg);

       public:
        explicit Device(const bool useDevice);
        explicit Device(const int deviceID);
        Device(const Device& other)     = default;   // copy constructor
        Device(Device&& other) noexcept = default;   // move constructor
        ~Device();
        Device& operator=(Device&& other) noexcept;         // move assignment
        Device& operator=(const Device& other) = default;   // copy assignment

        [[nodiscard]] bool isDeviceUsed() const;

        void checkErrors();
        void checkErrors(const std::string& msg);

        template <typename T>
        deviceError_t deviceMalloc(T** ptr, const size_t size);

        template <typename T>
        deviceError_t deviceFree(T* ptr);

        template <typename T>
        deviceError_t deviceMemcpyTo(T* dst, const std::vector<T>& src);
        template <typename T>
        deviceError_t deviceMemcpyTo(T* dst, const T* src, const size_t size);
        template <typename T>
        deviceError_t deviceMemcpyFrom(std::vector<T>& dst, const T* src);
        template <typename T>
        deviceError_t deviceMemcpyFrom(T* dst, const T* src, const size_t size);

        template <typename T>
        deviceError_t deviceMemcpyToAsync(T* dst, const std::vector<T>& src);
        template <typename T>
        deviceError_t deviceMemcpyToAsync(
            T*           dst,
            const T*     src,
            const size_t size
        );
        template <typename T>
        deviceError_t deviceMemcpyFromAsync(std::vector<T>& dst, const T* src);
        template <typename T>
        deviceError_t deviceMemcpyFromAsync(
            T*           dst,
            const T*     src,
            const size_t size
        );
    };

}   // namespace device

#include "device.tpp.hpp"   // IWYU pragma: keep DO NOT MOVE THIS LINE!!!

#endif   // __DEVICE_HPP__