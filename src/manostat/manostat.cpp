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

#include "manostat.hpp"

#include <functional>   // for function

#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox

using namespace manostat;

/**
 * @brief calculate the pressure of the system
 *
 * @param physicalData
 */
void Manostat::calculatePressure(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    auto       ekinVirial  = physicalData.getKineticEnergyVirialVector();
    auto       forceVirial = physicalData.getVirial();
    const auto volume      = box.getVolume();

    ekinVirial  = box.getBox().transformIntoOrthogonalSpace(ekinVirial);
    forceVirial = box.getBox().transformIntoOrthogonalSpace(forceVirial);

    _pressureTensor = (2.0 * ekinVirial + forceVirial) / volume *
                      constants::_PRESSURE_FACTOR_;

    _pressure = trace(_pressureTensor) / 3.0;

    physicalData.setPressure(_pressure);
}

/**
 * @brief apply dummy manostat for NVT ensemble
 *
 * @param physicalData
 */
void Manostat::applyManostat(
    simulationBox::SimulationBox &box,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("Calc Pressure");
    calculatePressure(box, physicalData);
    stopTimingsSection("Calc Pressure");
}