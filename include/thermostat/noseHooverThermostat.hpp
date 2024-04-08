/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _NOSE_HOOVER_THERMOSTAT_HPP_

#define _NOSE_HOOVER_THERMOSTAT_HPP_

#include "thermostat.hpp"

#include <vector>   // for std::vector

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