#include "device.hpp"

#include "exceptions.hpp"

using namespace device;

/**
 * @brief function to check for errors in the device API
 *
 * @note This function is used to check if in the latest device API calls an
 * error occurred.
 *
 * @param msg optional message to print in case of error. Default is no
 * additional message. The msg parameter will be inserted in a string "Error in
 * " + msg + ":\n" + error messages.
 */
void Device::checkErrors(const std::string& msg = "")
{
    std::string _msg = msg;
    if (_errorMsgs.size() > 0)
    {
        if (_msg.empty())
            _msg = "Device API call";

        std::string errorMsg = "Error in " + _msg + ":\n";

        for (const auto& e : _errorMsgs) errorMsg = errorMsg + e + "\n";

        throw customException::DeviceException(errorMsg);
    }
}