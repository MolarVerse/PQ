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
    std::string _jobtype;

public:
    // standard getter and setters
    std::string getStartFilename() const { return _startFilename; };
    void setStartFilename(std::string_view startFilename) { _startFilename = startFilename; };

    std::string getMoldescriptorFilename() const { return _moldescriptorFilename; };
    void setMoldescriptorFilename(std::string_view moldescriptorFilename) { _moldescriptorFilename = moldescriptorFilename; };

    std::string getGuffPath() const { return _guffPath; };
    void setGuffPath(std::string_view guffPath) { _guffPath = guffPath; };

    std::string getJobtype() const { return _jobtype; };
    void setJobtype(std::string_view jobtype) { _jobtype = jobtype; };
};

#endif