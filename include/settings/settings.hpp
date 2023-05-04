#ifndef _SETTINGS_H_

#define _SETTINGS_H_

#include <vector>

#include "timings.hpp"
#include "jobtype.hpp"
#include "output.hpp"

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
    JobType _jobType;
    std::vector<Output> _output = {StdoutOutput()};

    std::string getStartFilename() const { return _startFilename; };
    std::string setStartFilename(std::string_view startFilename) { return _startFilename = startFilename; };
};

#endif