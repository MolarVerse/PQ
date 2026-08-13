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

#include "mmmdEngine.hpp"

using namespace engine;

/**
 * @brief calculate MM forces
 *
 */
void MMMDEngine::calculateForces()
{
    _cellList->updateCellList(*_simulationBox);

    _potential->calculateForces(*_simulationBox, *_physicalData, *_cellList);

    _interWater->calculate(
        *_simulationBox,
        *_physicalData,
        _potential->getCoulombPotSharedPtr(),
        *_cellList
    );

    _intraNonBonded->calculate(*_simulationBox, *_physicalData);

    _virial->calculateVirial(*_simulationBox, *_physicalData);

    _forceField->calculateBondedInteractions(*_simulationBox, *_physicalData);

    _intraWater->calculate(*_simulationBox, *_physicalData);
}
