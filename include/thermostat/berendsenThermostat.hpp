#ifndef _BERENDSEN_THERMOSTAT_HPP_

#define _BERENDSEN_THERMOSTAT_HPP_

#include "thermostat.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace thermostat
{
    /**
     * @class BerendsenThermostat
     *
     * @brief BerendsenThermostat is a class for the Berendsen thermostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class BerendsenThermostat : public Thermostat
    {
      private:
        double _tau;

      public:
        BerendsenThermostat() = default;
        explicit BerendsenThermostat(const double targetTemp, const double tau) : Thermostat(targetTemp), _tau(tau) {}

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] double getTau() const { return _tau; }
        void                 setTau(const double tau) { _tau = tau; }
    };

}   // namespace thermostat

#endif   // _BERENDSEN_THERMOSTAT_HPP_