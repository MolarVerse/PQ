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

#include "langevinThermostat.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _KG_TO_GRAM_
#include "constants/natureConstants.hpp"     // for _UNIVERSAL_GAS_CONSTANT_
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "thermostatSettings.hpp"            // for ThermostatType
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for operator*, Vec3D

using thermostat::LangevinThermostat;
using namespace constants;
using namespace physicalData;
using namespace simulationBox;
using namespace settings;
using namespace linearAlgebra;

/**
 * @brief apply Langevin thermostat
 *
 * @details calculates the friction and random factor for each atom and applies
 * the Langevin thermostat to the velocities
 *
 * @param simBox
 */
void LangevinThermostat::applyLangevin(SimulationBox &simBox)
{
    auto applyFriction = [this](auto &atom)
    {
        const auto mass     = atom->getMass();
        const auto timeStep = TimingsSettings::getTimeStep();

        const auto propagationFactor = 0.5 * timeStep * _FS_TO_S_ / mass;

        const Vec3D randomFactor = {
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator)
        };

        const auto velocity = atom->getVelocity();
        auto       dv       = -propagationFactor * _friction * mass * velocity;

        dv += propagationFactor * _sigma * std::sqrt(mass) * randomFactor;

        atom->addVelocity(dv);
    };

    std::ranges::for_each(simBox.getAtoms(), applyFriction);
}