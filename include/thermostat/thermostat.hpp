#ifndef _THEMOSTAT_H_

#define _THEMOSTAT_H_

#include "simulationBox.hpp"
#include "physicalData.hpp"

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
    explicit Thermostat(const double targetTemperature) : _targetTemperature(targetTemperature) {}
    virtual ~Thermostat() = default;

    void calculateTemperature(SimulationBox &, PhysicalData &);

    virtual void applyThermostat(SimulationBox &, PhysicalData &);

    // standard getters and setters
    void setTimestep(const double timestep) { _timestep = timestep; }
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
    explicit BerendsenThermostat(const double targetTemperature, const double tau) : Thermostat(targetTemperature), _tau(tau) {}
    // standard getters and setters
    [[nodiscard]] double getTau() const { return _tau; }
    void setTau(const double tau) { _tau = tau; }

    void applyThermostat(SimulationBox &, PhysicalData &) override;
};

#endif // _THEMOSTAT_H_