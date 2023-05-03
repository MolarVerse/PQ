#ifndef _SETTINGS_H_

#define _SETTINGS_H_

#include "timings.hpp"
#include "jobtype.hpp"

/**
 * @class Settings
 *
 * @brief Stores the settings of the simulation
 *
 */
class Settings
{
public:
    Settings() = default;
    ~Settings() = default;

    Timings _timings;
    JobType _jobType;
};

#endif