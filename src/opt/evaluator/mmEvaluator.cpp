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

#include "mmEvaluator.hpp"

#include "intraNonBonded.hpp"
#include "settings.hpp"

namespace opt
{

    /**
     * @brief clones the evaluator
     *
     */
    std::shared_ptr<Evaluator> MMEvaluator::clone() const
    {
        return std::make_shared<MMEvaluator>(*this);
    }

    /**
     * @brief update forces
     *
     */
    void MMEvaluator::evaluate(
        physicalData::PhysicalData&     physicalData,
        molsys::SimulationBox&          simulationBox,
        intraNonBonded::IntraNonBonded& intraNonBonded
    )
    {
        simulationBox.updateOldForces();
        simulationBox.resetForces();

        _getCellList().updateCellList(
            simulationBox,
            _getSettings().potentialSettings.getCoulombRadiusCutoff(),
            _getSettings().generalSettings
        );

        _getPotential().calculateForces(
            simulationBox,
            physicalData,
            _getCellList(),
            _getSettings().waterModelSettings,
            _getSettings().potentialSettings
        );

        intraNonBonded.calculate(
            simulationBox,
            physicalData,
            _getSettings().potentialSettings
        );

        _getForceField().calculateBondedInteractions(
            simulationBox,
            physicalData,
            _getSettings().potentialSettings
        );
    }

}   // namespace opt
