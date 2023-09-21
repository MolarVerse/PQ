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
        std::vector<double> _chi{0.0};    // in kcal/mol s
        std::vector<double> _zeta{0.0};   // unitless

        double _couplingFrequency = 0.0;   // in 1/s

      public:
        NoseHooverThermostat() = default;
        explicit NoseHooverThermostat(const double               targetTemp,
                                      const std::vector<double> &chi,
                                      const std::vector<double> &zeta,
                                      const double               couplingFrequency)
            : Thermostat(targetTemp), _chi(chi), _zeta(zeta), _couplingFrequency(couplingFrequency){};

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;
        void applyThermostatOnForces(simulationBox::SimulationBox &) override;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::vector<double> getChi() const { return _chi; }
        [[nodiscard]] std::vector<double> getZeta() const { return _zeta; }
        [[nodiscard]] double              getCouplingFrequency() const { return _couplingFrequency; }

        /***************************
         * standard setter methods *
         ***************************/

        void setChi(const unsigned int index, const double chi) { _chi[index] = chi; }
        void setChi(const std::vector<double> &chi) { _chi = chi; }

        void setZeta(const unsigned int index, const double zeta) { _zeta[index] = zeta; }
        void setZeta(const std::vector<double> &zeta) { _zeta = zeta; }

        void setCouplingFrequency(const double couplingFrequency) { _couplingFrequency = couplingFrequency; }
    };
}   // namespace thermostat

#endif   // _NOSE_HOOVER_THERMOSTAT_HPP_