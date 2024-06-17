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

/**
 * @brief set the potential as shared pointer
 *
 * @param potential - potential::Potential&
 */
void Evaluator::setPotential(const potential::Potential& potential)
{
    _potential = potential.clone();
}

/**
 * @brief set the cell list as shared pointer
 *
 * @param cellList - simulationBox::CellList&
 */
void Evaluator::setCellList(const simulationBox::CellList& cellList)
{
    _cellList = cellList.clone();
}

/**
 * @brief set the simulation box as shared pointer
 *
 * @param simulationBox - simulationBox::SimulationBox&
 */
void Evaluator::setSimulationBox(
    const simulationBox::SimulationBox& simulationBox
)
{
    _simulationBox = simulationBox.clone();
}

/**
 * @brief set the constraints as shared pointer
 *
 * @param constraints - constraints::Constraints&
 */
void Evaluator::setConstraints(const constraints::Constraints& constraints)
{
    _constraints = constraints.clone();
}

/**
 * @brief set the physical data as shared pointer
 *
 * @param physicalData - physicalData::PhysicalData&
 */
void Evaluator::setPhysicalData(const physicalData::PhysicalData& physicalData)
{
    _physicalData = physicalData.clone();
}

/**
 * @brief set the old physical data as shared pointer
 *
 * @param physicalData - physicalData::PhysicalData&
 */
void Evaluator::setPhysicalDataOld(
    const physicalData::PhysicalData& physicalData
)
{
    _physicalDataOld = physicalData.clone();
}

/**
 * @brief set the force field as shared pointer
 *
 * @param forceField - forceField::ForceField&
 */
void Evaluator::setForceField(const forceField::ForceField& forceField)
{
    _forceField = forceField.clone();
}

/**
 * @brief set the intra non bonded as shared pointer
 *
 * @param intraNonBonded - intraNonBonded::IntraNonBonded&
 */
void Evaluator::setIntraNonBonded(
    const intraNonBonded::IntraNonBonded& intraNonBonded
)
{
    _intraNonBonded = intraNonBonded.clone();
}

/**
 * @brief set the virial as shared pointer
 *
 * @param virial - virial::Virial&
 */
void Evaluator::setVirial(const virial::Virial& virial)
{
    _virial = virial.clone();
}