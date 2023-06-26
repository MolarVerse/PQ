#ifndef _THEMOSTAT_H_

#define _THEMOSTAT_H_

#include "physicalData.hpp"
#include "simulationBox.hpp"

/**
 * @namespace thermostat
 *
 */
namespace thermostat
{
    class Thermostat;
    class BerendsenThermostat;
}   // namespace thermostat

/**
 * @class Thermostat
 *
 * @brief Thermostat is a base class for all thermostats
 *
 */
class thermostat::Thermostat
{
  protected:
    double _temperature;
    double _targetTemperature;
    double _timestep;

  public:
    Thermostat() = default;
    explicit Thermostat(const double targetTemperature) : _targetTemperature(targetTemperature) {}
    virtual ~Thermostat() = default;

    virtual void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);

    void setTimestep(const double timestep) { _timestep = timestep; }
};

/**
 * @class BerendsenThermostat
 *
 * @brief BerendsenThermostat is a class for Berendsen thermostat
 *
 */
class thermostat::BerendsenThermostat : public thermostat::Thermostat
{
  private:
    double _tau;

  public:
    BerendsenThermostat() = default;
    explicit BerendsenThermostat(const double targetTemperature, const double tau) : Thermostat(targetTemperature), _tau(tau) {}

    void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

    double getTau() const { return _tau; }
    void   setTau(const double tau) { _tau = tau; }
};

#endif   // _THEMOSTAT_H_