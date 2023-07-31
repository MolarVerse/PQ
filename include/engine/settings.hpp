#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

#include "defaults.hpp"
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
    // resetKineticsSettings for later setup
    size_t _nScale = 0;
    size_t _fScale = 0;
    size_t _nReset = 0;
    size_t _fReset = 0;

    // filenames and paths for later setup
    std::string _startFilename;
    std::string _moldescriptorFilename = defaults::_MOLDESCRIPTOR_FILENAME_DEFAULT_;   // for backward compatibility
    std::string _guffPath              = ".";                                          // not backward compatible
    std::string _topologyFilename      = "";

    std::string _jobtype;

    // thermostat settings for later setup
    std::pair<bool, std::string> _thermostat;   // pair.first = check if thermostat was set
    std::pair<bool, double>      _temperature;
    std::pair<bool, double>      _relaxationTime = std::make_pair(false, 0.1);   // pay attention here default value in ps

    // manostat settings for later setup
    std::pair<bool, std::string> _manostat;   // pair.first = check if thermostat was set
    std::pair<bool, double>      _pressure;
    std::pair<bool, double>      _tauManostat = std::make_pair(false, 1.0);   // pay attention here default value in ps

    // shake settings for later setup
    double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;    // 1e-8
    size_t _shakeMaxIter    = defaults::_SHAKE_MAX_ITER_DEFAULT_;     // 20
    double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;   // 1e-8
    size_t _rattleMaxIter   = defaults::_RATTLE_MAX_ITER_DEFAULT_;    // 20

    // coulomb long range settings for later setup
    std::string _coulombLongRangeType = defaults::_COULOMB_LONG_RANGE_TYPE_DEFAULT_;   // none
    double      _wolfParameter        = defaults::_WOLF_PARAMETER_DEFAULT_;            // 0.25

  public:
    /********************
     * standard getters *
     ********************/

    std::string getStartFilename() const { return _startFilename; }
    std::string getMoldescriptorFilename() const { return _moldescriptorFilename; }
    std::string getGuffPath() const { return _guffPath; }
    std::string getTopologyFilename() const { return _topologyFilename; }

    std::string getJobtype() const { return _jobtype; }

    std::string getThermostat() const { return _thermostat.second; }
    bool        getThermostatSet() const { return _thermostat.first; }
    double      getTemperature() const { return _temperature.second; }
    bool        getTemperatureSet() const { return _temperature.first; }
    double      getRelaxationTime() const { return _relaxationTime.second; }
    bool        getRelaxationTimeSet() const { return _relaxationTime.first; }

    std::string getManostat() const { return _manostat.second; }
    bool        getManostatSet() const { return _manostat.first; }
    double      getPressure() const { return _pressure.second; }
    bool        getPressureSet() const { return _pressure.first; }
    double      getTauManostat() const { return _tauManostat.second; }
    bool        getTauManostatSet() const { return _tauManostat.first; }

    size_t getNScale() const { return _nScale; }
    size_t getFScale() const { return _fScale; }
    size_t getNReset() const { return _nReset; }
    size_t getFReset() const { return _fReset; }

    double getShakeTolerance() const { return _shakeTolerance; }
    size_t getShakeMaxIter() const { return _shakeMaxIter; }
    double getRattleTolerance() const { return _rattleTolerance; }
    size_t getRattleMaxIter() const { return _rattleMaxIter; }

    std::string getCoulombLongRangeType() const { return _coulombLongRangeType; }
    double      getWolfParameter() const { return _wolfParameter; }

    /********************
     * standard setters *
     ********************/

    void setStartFilename(const std::string_view startFilename) { _startFilename = startFilename; }
    void setMoldescriptorFilename(const std::string_view filename) { _moldescriptorFilename = filename; }
    void setGuffPath(const std::string_view guffPath) { _guffPath = guffPath; }
    void setTopologyFilename(const std::string_view topologyFilename) { _topologyFilename = topologyFilename; }

    void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }

    void setThermostat(const std::string_view thermostat) { _thermostat = std::make_pair(true, thermostat); }
    void setTemperature(const double temperature);
    void setRelaxationTime(const double relaxationTime);

    void setManostat(const std::string_view manostat) { _manostat = std::make_pair(true, manostat); }
    void setPressure(const double pressure) { _pressure = std::make_pair(true, pressure); }
    void setTauManostat(const double tauManostat);

    void setNScale(const size_t nScale) { _nScale = nScale; }
    void setFScale(const size_t fScale) { _fScale = fScale; }
    void setNReset(const size_t nReset) { _nReset = nReset; }
    void setFReset(const size_t fReset) { _fReset = fReset; }

    void setShakeTolerance(const double shakeTolerance) { _shakeTolerance = shakeTolerance; }
    void setShakeMaxIter(const size_t shakeMaxIter) { _shakeMaxIter = shakeMaxIter; }
    void setRattleTolerance(const double rattleTolerance) { _rattleTolerance = rattleTolerance; }
    void setRattleMaxIter(const size_t rattleMaxIter) { _rattleMaxIter = rattleMaxIter; }

    void setCoulombLongRangeType(const std::string_view coulombLongRangeType) { _coulombLongRangeType = coulombLongRangeType; }
    void setWolfParameter(const double wolfParameter) { _wolfParameter = wolfParameter; }
};

#endif   // _SETTINGS_HPP_