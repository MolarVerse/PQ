#ifndef _SETTINGS_H_

#define _SETTINGS_H_

#include <vector>
#include <string>

#include "timings.hpp"

/**
 * @class Settings
 *
 * @brief
 *  Stores the settings of the simulation
 *  Additionally it stores all information needed for later setup of the simulation
 *
 */
class Settings
{
private:
    std::string _startFilename;
    std::string _moldescriptorFilename = "moldescriptor.dat"; // for backward compatibility
    std::string _guffPath = ".";                              // not backward compatible

    std::string _jobtype;

    std::pair<bool, std::string> _thermostat; // pair.first = check if thermostat was set
    std::pair<bool, double> _temperature;
    std::pair<bool, double> _relaxationTime = std::make_pair(false, 0.1); // pay attention here default value in ps

public:
    // standard getter and setters
    std::string getStartFilename() const { return _startFilename; };
    void setStartFilename(const std::string_view startFilename) { _startFilename = startFilename; };

    std::string getMoldescriptorFilename() const { return _moldescriptorFilename; };
    void setMoldescriptorFilename(const std::string_view moldescriptorFilename) { _moldescriptorFilename = moldescriptorFilename; };

    std::string getGuffPath() const { return _guffPath; };
    void setGuffPath(const std::string_view guffPath) { _guffPath = guffPath; };

    std::string getJobtype() const { return _jobtype; };
    void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; };

    [[nodiscard]] bool getThermostatSet() const { return _thermostat.first; };
    std::string getThermostat() const { return _thermostat.second; };
    void setThermostat(std::string_view thermostat) { _thermostat = std::make_pair(true, thermostat); };

    [[nodiscard]] bool getTemperatureSet() const { return _temperature.first; };
    [[nodiscard]] double getTemperature() const { return _temperature.second; };
    void setTemperature(double temperature);

    [[nodiscard]] bool getRelaxationTimeSet() const { return _relaxationTime.first; };
    [[nodiscard]] double getRelaxationTime() const { return _relaxationTime.second; };
    void setRelaxationTime(double relaxationTime);
};

#endif