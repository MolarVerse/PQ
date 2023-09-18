#ifndef _NOSE_HOOVER_THERMOSTAT_HPP_

#define _NOSE_HOOVER_THERMOSTAT_HPP_

#include "thermostat.hpp"

#include <vector>   // for std::vector

namespace thermostat
{
    /**
     * @class NoseHooverThermostat
     *
     * @brief this class implements the Nose-Hoover thermostat
     *
     */
    class NoseHooverThermostat : public Thermostat
    {
      private:
        std::vector<double> _chi{0.0};
        std::vector<double> _zeta{0.0};

        double _omegaFactor = 0.0;

      public:
        NoseHooverThermostat() = default;
        explicit NoseHooverThermostat(const double targetTemp, const std::vector<double> &chi)
            : Thermostat(targetTemp), _chi(chi){};

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override{};

        [[nodiscard]] std::vector<double> getChi() const { return _chi; }
        void                              setChi(const std::vector<double> &chi) { _chi = chi; }
    };
}   // namespace thermostat

#endif   // _NOSE_HOOVER_THERMOSTAT_HPP_