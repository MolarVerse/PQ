#ifndef _VELOCITY_RESCALING_THERMOSTAT_HPP_

#define _VELOCITY_RESCALING_THERMOSTAT_HPP_

#include "thermostat.hpp"

#include <random>   // for std::random_device, std::mt19937

namespace thermostat
{
    class VelocityRescalingThermostat : public Thermostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _tau = 0.0;

      public:
        VelocityRescalingThermostat() = default;
        explicit VelocityRescalingThermostat(const double targetTemp, const double tau) : Thermostat(targetTemp), _tau(tau) {}
        VelocityRescalingThermostat(const VelocityRescalingThermostat &);

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] double getTau() const { return _tau; }
        void                 setTau(const double tau) { _tau = tau; }
    };
}   // namespace thermostat

#endif   // _VELOCITY_RESCALING_THERMOSTAT_HPP_