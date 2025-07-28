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

#include "evaluator.hpp"

#include "celllist.hpp"
#include "constraints.hpp"
#include "forceFieldClass.hpp"
#include "intraNonBonded.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "virial.hpp"

using namespace opt;
using namespace potential;
using namespace simulationBox;
using namespace physicalData;
using namespace forceField;
using namespace intraNonBonded;
using namespace virial;
using namespace constraints;

/***************************
 * standard setter methods *
 ***************************/

/**
 * @brief set the potential as shared pointer
 *
 * @param potential - std::shared_ptr<Potential>
 */
void Evaluator::setPotential(const std::shared_ptr<Potential> potential)
{
    _potential = potential;
}

/**
 * @brief set the cell list as shared pointer
 *
 * @param cellList - std::shared_ptr<CellList>
 */
void Evaluator::setCellList(const std::shared_ptr<CellList> cellList)
{
    _cellList = cellList;
}

/**
 * @brief set the simulation box as shared pointer
 *
 * @param simulationBox - std::shared_ptr<SimulationBox>
 */
void Evaluator::setSimulationBox(
    const std::shared_ptr<SimulationBox> simulationBox
)
{
    _simulationBox = simulationBox;
}

/**
 * @brief set the constraints as shared pointer
 *
 * @param constraints - std::shared_ptr<Constraints>
 */
void Evaluator::setConstraints(const std::shared_ptr<Constraints> constraints)
{
    _constraints = constraints;
}

/**
 * @brief set the physical data as shared pointer
 *
 * @param physicalData - std::shared_ptr<PhysicalData>
 */
void Evaluator::setPhysicalData(
    const std::shared_ptr<PhysicalData> physicalData
)
{
    _physicalData = physicalData;
}

/**
 * @brief set the old physical data as shared pointer
 *
 * @param physicalData - std::shared_ptr<PhysicalData>
 */
void Evaluator::setPhysicalDataOld(
    const std::shared_ptr<PhysicalData> physicalData
)
{
    _physicalDataOld = physicalData;
}

/**
 * @brief set the force field as shared pointer
 *
 * @param forceField - std::shared_ptr<ForceField>
 */
void Evaluator::setForceField(const std::shared_ptr<ForceField> forceField)
{
    _forceField = forceField;
}

/**
 * @brief set the intra non bonded as shared pointer
 *
 * @param intraNonBonded - std::shared_ptr<IntraNonBonded>
 */
void Evaluator::setIntraNonBonded(
    const std::shared_ptr<IntraNonBonded> intraNonBonded
)
{
    _intraNonBonded = intraNonBonded;
}

/**
 * @brief set the virial as shared pointer
 *
 * @param virial - std::shared_ptr<Virial>
 */
void Evaluator::setVirial(const std::shared_ptr<Virial> virial)
{
    _virial = virial;
}