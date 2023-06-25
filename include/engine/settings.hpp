#ifndef _SETTINGS_H_

#define _SETTINGS_H_

#include "timings.hpp"

#include <string>
#include <vector>

namespace settings
{
    class Settings;
}

/**
 * @class Settings
 *
 * @brief
 *  Stores the settings of the simulation
 *  Additionally it stores all information needed for later setup of the simulation
 *
 */
class settings::Settings
{
  private:
    std::string _startFilename;
    std::string _moldescriptorFilename = "moldescriptor.dat";   // for backward compatibility
    std::string _guffPath              = ".";                   // not backward compatible

    std::string _jobtype;

    std::pair<bool, std::string> _thermostat;                                    // pair.first = check if thermostat was set
    std::pair<bool, double>      _temperature;
    std::pair<bool, double>      _relaxationTime = std::make_pair(false, 0.1);   // pay attention here default value in ps

    std::pair<bool, std::string> _manostat;                                      // pair.first = check if thermostat was set
    std::pair<bool, double>      _pressure;
    std::pair<bool, double>      _tauManostat = std::make_pair(false, 1.0);      // pay attention here default value in ps

  public:
    /********************
     * standard getters *
     ********************/

    std::string getStartFilename() const { return _startFilename; }
    std::string getMoldescriptorFilename() const { return _moldescriptorFilename; }
    std::string getGuffPath() const { return _guffPath; }
    std::string getJobtype() const { return _jobtype; }
    std::string getThermostat() const { return _thermostat.second; }
    std::string getManostat() const { return _manostat.second; }
    bool        getThermostatSet() const { return _thermostat.first; }
    bool        getTemperatureSet() const { return _temperature.first; }
    bool        getRelaxationTimeSet() const { return _relaxationTime.first; }
    bool        getManostatSet() const { return _manostat.first; }
    bool        getPressureSet() const { return _pressure.first; }
    bool        getTauManostatSet() const { return _tauManostat.first; }
    double      getTemperature() const { return _temperature.second; }
    double      getRelaxationTime() const { return _relaxationTime.second; }
    double      getPressure() const { return _pressure.second; }
    double      getTauManostat() const { return _tauManostat.second; }

    /********************
     * standard setters *
     ********************/

    void setStartFilename(const std::string_view startFilename) { _startFilename = startFilename; }
    void setMoldescriptorFilename(const std::string_view filename) { _moldescriptorFilename = filename; }
    void setGuffPath(const std::string_view guffPath) { _guffPath = guffPath; }
    void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }
    void setThermostat(std::string_view thermostat) { _thermostat = std::make_pair(true, thermostat); }
    void setTemperature(double temperature);
    void setRelaxationTime(double relaxationTime);
    void setManostat(std::string_view manostat) { _manostat = std::make_pair(true, manostat); }
    void setPressure(double pressure) { _pressure = std::make_pair(true, pressure); }
    void setTauManostat(double tauManostat);
};

#endif