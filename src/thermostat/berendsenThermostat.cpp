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

#include "berendsenThermostat.hpp"

#include "atom.hpp"              // for Atom
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <cmath>    // for sqrt
#include <memory>   // for __shared_ptr_access, shared_ptr
#include <vector>   // for vector

using thermostat::BerendsenThermostat;

/**
 * @brief apply thermostat - Berendsen
 *
 * @link https://doi.org/10.1063/1.448118
 *
 * @param simulationBox
 * @param physicalData
 */
void BerendsenThermostat::applyThermostat(simulationBox::SimulationBox &simulationBox, physicalData::PhysicalData &physicalData)
{
    physicalData.calculateTemperature(simulationBox);

    _temperature = physicalData.getTemperature();

    const auto berendsenFactor =
        ::sqrt(1.0 + settings::TimingsSettings::getTimeStep() / _tau * (_targetTemperature / _temperature - 1.0));

    for (const auto &atom : simulationBox.getAtoms())
        atom->scaleVelocity(berendsenFactor);

    physicalData.setTemperature(_temperature * berendsenFactor * berendsenFactor);
}