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

#include "device.hpp"

#include <format>
#include <iostream>

#include "deviceConfig.hpp"
#include "exceptions.hpp"

using namespace device;

/**
 * @brief Construct a new Device:: Device object
 *
 * @details This constructor is used to initialize the device object. The device
 * object is used to handle all device API calls.
 * The constructor initializes the following variables:
 *      - _deviceID: the device ID of the device
 *      - _deviceCount: the number of devices in the system
 *      - _deviceProp: the device properties of the device
 *      - _dataStream: the data stream of the device
 *      - _computeStream: the compute stream of the device
 *
 *
 * @note If the device ID is not given, the constructor will get the device ID
 * from the device API. If the device ID is given, the constructor will use the
 * given device ID. The constructor will also get the device count from the
 * device API.
 *
 * @throw DeviceException if an error occurred during the device API calls or
 * if the device ID is out of range.
 *
 * @param useDevice if true the device will be used, if false the device will
 * not be used
 */
Device::Device(const bool useDevice)
{
    if (!useDevice)
    {
        _useDevice = false;
        return;
    }

    int                 deviceID = -1;
    const deviceError_t error    = __getDevice(&deviceID);

    addDeviceError(error, "Getting the device ID");

    *this = Device(deviceID);
}

Device::Device(const int deviceID)
    : _useDevice(true), _deviceID(deviceID), _deviceCount(0)
{
    deviceError_t error = __getDeviceCount(&_deviceCount);
    addDeviceError(error, "Getting the device count");

    if (_deviceID >= _deviceCount)
        _errorMsgs.push_back(std::format(
            "The device ID is out of range. The device ID is {} and the "
            "device count is {}",
            _deviceID,
            _deviceCount
        ));

    error = __getDeviceProperties(&_deviceProp, _deviceID);
    addDeviceError(error, "Getting the device properties");

    error = __setDevice(_deviceID);
    addDeviceError(
        error,
        std::format("Setting the device with the device ID {}", _deviceID)
    );

    error = __deviceStreamCreate(&_dataStream);
    addDeviceError(error, "Creating the data stream");

    error = __deviceStreamCreate(&_computeStream);
    addDeviceError(error, "Creating the compute stream");

    checkErrors("Device initialization");
}

/**
 * @brief Destroy the Device:: Device object
 *
 * @details This destructor is used to destroy the device object. The destructor
 * will free the data stream and the compute stream.
 */
Device::~Device()
{
    // TODO: check why this gives segfault

    // if (_dataStream != nullptr)
    // {
    //     const auto error = __deviceStreamDestroy(_dataStream);
    //     addDeviceError(error, "Destroying the data stream");
    // }

    // if (_computeStream != nullptr)
    // {
    //     const auto error = __deviceStreamDestroy(_computeStream);
    //     addDeviceError(error, "Destroying the compute stream");
    // }

    // to avoid destroying thrown exceptions in the destructor during stack
    // unwinding
    if (_uncaughtExceptions != std::uncaught_exceptions())
        return;

    // To not throw an uncaught exception in the destructor
    try
    {
        checkErrors("Device destruction");
    }
    catch (const customException::DeviceException& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

/**
 * @brief Move assignment operator
 *
 * @details This move assignment operator is used to move the device object. The
 * move assignment operator will move the device object to the new device
 * object.
 *
 * @param other the device object that should be moved
 *
 * @return Device& the moved device object
 */
Device& Device::operator=(Device&& other) noexcept
{
    if (this != &other)
    {
        _useDevice     = other._useDevice;
        _deviceID      = other._deviceID;
        _deviceCount   = other._deviceCount;
        _deviceProp    = other._deviceProp;
        _dataStream    = other._dataStream;
        _computeStream = other._computeStream;
        _errorMsgs     = std::move(other._errorMsgs);

        // to avoid destroying the streams in the destructor
        other._dataStream    = nullptr;
        other._computeStream = nullptr;
    }

    return *this;
}

/**
 * @brief function to check if the device is used
 *
 * @details This function is used to check if the device is used. The function
 * will return true if the device is used and false if the device is not used.
 *
 * @return true if the device is used
 * @return false if the device is not used
 */
[[nodiscard]] bool Device::isDeviceUsed() const { return _useDevice; }

/**
 * @brief function to add an error message to the error message list
 *
 * @details This function is used to add an error message to the error message
 * list. The error message list is used to store all error messages that
 * occurred during the device API calls.
 *
 * @throw DeviceException if the error code is not __deviceSuccess__
 *
 * @param error the error code that occurred
 * @param msg the message that should be added to the error message list
 */
void device::Device::addDeviceError(
    const deviceError_t error,
    const std::string&  msg
)
{
    if (error != __deviceSuccess__)
        _errorMsgs.push_back(
            msg + " failed with the following error:\n\n" +
            __deviceGetErrorString(error)
        );
}

/**
 * @brief function to check for errors in the device API
 *
 * @note This function is used to check if in the latest device API calls an
 * error occurred.
 *
 * @param msg optional message to print in case of error. Default is no
 * additional message. The msg parameter will be inserted in a string "Error
 * in " + msg + ":\n" + error messages.
 */
void Device::checkErrors() { checkErrors("Device API call"); }

void Device::checkErrors(const std::string& msg)
{
    if (!_errorMsgs.empty())
    {
        std::string errorMsg = "Error in " + msg + ":\n\n";

        for (const auto& e : _errorMsgs)
            errorMsg = errorMsg + e + "\n";

        throw customException::DeviceException(errorMsg);
    }
}