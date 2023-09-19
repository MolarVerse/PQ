#ifndef _THERMOSTAT_HPP_

#define _THERMOSTAT_HPP_

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

/**
 * @namespace thermostat
 */
namespace thermostat
{
    /**
     * @class Thermostat
     *
     * @brief Thermostat is a base class for all thermostats
     *
     * @details it provides a dummy function applyThermostat() which does only calculate the temperature
     *
     */
    class Thermostat
    {
      protected:
        double _temperature       = 0.0;
        double _targetTemperature = 0.0;

      public:
        Thermostat() = default;
        explicit Thermostat(const double targetTemperature) : _targetTemperature(targetTemperature) {}
        virtual ~Thermostat() = default;

        virtual void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void applyThermostatHalfStep(simulationBox::SimulationBox &, physicalData::PhysicalData &){};
        virtual void applyThermostatOnForces(simulationBox::SimulationBox &){};
    };

}   // namespace thermostat

#endif   // _THERMOSTAT_HPP_