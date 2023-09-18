#ifndef _LANGEVIN_THERMOSTAT_HPP_

#define _LANGEVIN_THERMOSTAT_HPP_

#include "thermostat.hpp"

#include <random>   // for std::random_device, std::mt19937

namespace thermostat
{

    class LangevinThermostat : public Thermostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _sigma = 0.0;

      public:
        LangevinThermostat() = default;
        explicit LangevinThermostat(const double targetTemperature);

        virtual void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void applyThermostatHalfStep(simulationBox::SimulationBox &, physicalData::PhysicalData &){};
    };

}   // namespace thermostat

#endif   // _LANGEVIN_THERMOSTAT_HPP_