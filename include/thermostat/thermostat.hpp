#ifndef _THEMOSTAT_H_

#define _THEMOSTAT_H_

#include "simulationBox.hpp"

/**
 * @class Thermostat
 *
 * @brief Thermostat is a base class for all thermostats
 *
 */
class Thermostat
{
protected:
    double _temperature;
    double _targetTemperature;
    double _timestep;

public:
    Thermostat() = default;
    explicit Thermostat(double targetTemperature) : _targetTemperature(targetTemperature) {}
    virtual ~Thermostat() = default;

    void calculateTemperature(const SimulationBox &);

    virtual void applyThermostat(SimulationBox &);

    // standard getters and setters
    double getTemperature() const { return _temperature; }
    void setTemperature(double temperature) { _temperature = temperature; }

    double getTargetTemperature() const { return _targetTemperature; }
    void setTargetTemperature(double targetTemperature) { _targetTemperature = targetTemperature; }

    double getTimestep() const { return _timestep; }
    void setTimestep(double timestep) { _timestep = timestep; }
};

/**
 * @class BerendsenThermostat
 *
 * @brief BerendsenThermostat is a class for Berendsen thermostat
 *
 */
class BerendsenThermostat : public Thermostat
{
private:
    double _tau;

public:
    BerendsenThermostat() = default;
    explicit BerendsenThermostat(double targetTemperature, double tau) : Thermostat(targetTemperature), _tau(tau) {}
    // standard getters and setters
    double getTau() const { return _tau; }
    void setTau(double tau) { _tau = tau; }

    void applyThermostat(SimulationBox &) override;
};

#endif // _THEMOSTAT_H_