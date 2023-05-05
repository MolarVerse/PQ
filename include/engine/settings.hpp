#ifndef _SETTINGS_H_

#define _SETTINGS_H_

#include <vector>
#include <string>

#include "timings.hpp"

/**
 * @class Settings
 *
 * @brief Stores the settings of the simulation
 *
 */
class Settings
{
private:
    std::string _startFilename;
    std::string _moldescriptorFilename = "moldescriptor.dat"; // for backward compatibility
    std::string _guffPath = ".";                              // not backward compatible

public:
    Settings() = default;
    ~Settings() = default;

    Timings _timings;

    std::string getStartFilename() const { return _startFilename; };
    std::string setStartFilename(std::string_view startFilename) { return _startFilename = startFilename; };

    std::string getMoldescriptorFilename() const { return _moldescriptorFilename; };
    std::string setMoldescriptorFilename(std::string_view moldescriptorFilename) { return _moldescriptorFilename = moldescriptorFilename; };

    std::string getGuffPath() const { return _guffPath; };
    std::string setGuffPath(std::string_view guffPath) { return _guffPath = guffPath; };
};

#endif