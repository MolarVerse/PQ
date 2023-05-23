#ifndef _THEMOSTAT_H_

#define _THEMOSTAT_H_

#include "simulationBox.hpp"
#include "outputData.hpp"

/**
 * @class Thermostat
 *
 * @brief Thermostat is a base class for all thermostats
 *
 */
class Thermostat
{
private:
    double _temperature;
    double _targetTemperature;

public:
    Thermostat() = default;
    explicit Thermostat(double targetTemperature) : _targetTemperature(targetTemperature) {}
    virtual ~Thermostat() = default;

    void calculateTemperature(const SimulationBox &, OutputData &);

    // standard getters and setters
    double getTemperature() const { return _temperature; }
    void setTemperature(double temperature) { _temperature = temperature; }

    double getTargetTemperature() const { return _targetTemperature; }
    void setTargetTemperature(double targetTemperature) { _targetTemperature = targetTemperature; }
};

#endif // _THEMOSTAT_H_