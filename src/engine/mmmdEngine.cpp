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
            get_simulationBox(),
            getSettings().potentialSettings.getCoulombRadiusCutOff()
        );

        getPotential()->calculateForces(
            get_simulationBox(),
            get_physicalData(),
            getCellList(),
            getSettings().waterModelSettings,
            getSettings().potentialSettings
        );

        _get_interWater().calculate(
            get_simulationBox(),
            get_physicalData(),
            getPotential()->getCoulombPotSharedPtr(),
            getCellList(),
            getSettings().potentialSettings
        );

        get_intraNonBonded().calculate(
            get_simulationBox(),
            get_physicalData(),
            getSettings().potentialSettings
        );

        const auto virial = virial::calculateVirial(get_simulationBox());
        get_physicalData().set_virial(virial);

        getForceField()->calculateBondedInteractions(
            get_simulationBox(),
            get_physicalData(),
            getSettings().potentialSettings
        );

        _get_intraWater().calculate(get_simulationBox(), get_physicalData());
    }

}   // namespace engine
