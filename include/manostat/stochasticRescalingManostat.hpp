/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#define _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#include "manostat.hpp"   // for Manostat

#include <random>   // for std::random_device, std::mt19937

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace manostat
{
    /**
     * @class StochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class StochasticRescalingManostat : public Manostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _tau;
        double _compressibility;

      public:
        StochasticRescalingManostat() = default;
        StochasticRescalingManostat(const StochasticRescalingManostat &);
        explicit StochasticRescalingManostat(const double targetPressure, const double tau, const double compressibility)
            : Manostat(targetPressure), _tau(tau), _compressibility(compressibility){};

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getTau() const { return _tau; }
    };

}   // namespace manostat

#endif   // _STOCHASTIC_RESCALING_MANOSTAT_HPP_