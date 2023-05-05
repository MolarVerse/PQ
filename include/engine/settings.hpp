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

public:
    Settings() = default;
    ~Settings() = default;

    Timings _timings;

    std::string getStartFilename() const { return _startFilename; };
    std::string setStartFilename(std::string_view startFilename) { return _startFilename = startFilename; };
};

#endif