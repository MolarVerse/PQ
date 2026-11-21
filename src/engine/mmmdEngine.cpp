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

#include "celllist.hpp"
#include "virial.hpp"

namespace engine
{

    /**
     * @brief calculate MM forces
     *
     */
    void MMMDEngine::calculateForces()
    {
        getCellList().updateCellList(
            getSimulationBox(),
            getSettings().potentialSettings.getCoulombRadiusCutOff()
        );

        getPotential()->calculateForces(
            getSimulationBox(),
            get_physicalData(),
            getCellList(),
            getSettings().waterModelSettings,
            getSettings().potentialSettings
        );

        _getInterWater().calculate(
            getSimulationBox(),
            get_physicalData(),
            getPotential()->getCoulombPotSharedPtr(),
            getCellList(),
            getSettings().potentialSettings
        );

        getIntraNonBonded()->calculate(
            getSimulationBox(),
            get_physicalData(),
            getSettings().potentialSettings
        );

        const auto virial = virial::calculateVirial(getSimulationBox());
        get_physicalData().setVirial(virial);

        getForceField()->calculateBondedInteractions(
            getSimulationBox(),
            get_physicalData(),
            getSettings().potentialSettings
        );

        _getIntraWater().calculate(getSimulationBox(), get_physicalData());
    }

}   // namespace engine
