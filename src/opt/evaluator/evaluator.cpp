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

using namespace opt;

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the potential
 *
 * @param potential
 */
void Evaluator::setPotential(
    const std::shared_ptr<potential::Potential> potential
)
{
    _potential = potential;
}

/**
 * @brief set the simulation box
 *
 * @param simBox
 */
void Evaluator::setSimulationBox(
    const std::shared_ptr<simulationBox::SimulationBox> simBox
)
{
    _simulationBox = simBox;
}

/**
 * @brief set the constraints
 *
 * @param constraints
 */
void Evaluator::setConstraints(
    const std::shared_ptr<constraints::Constraints> constraints
)
{
    _constraints = constraints;
}

/**
 * @brief set the cell list
 *
 * @param cellList
 */
void Evaluator::setCellList(
    const std::shared_ptr<simulationBox::CellList> cellList
)
{
    _cellList = cellList;
}

/**
 * @brief set the force field
 *
 * @param forceField
 */
void Evaluator::setForceField(
    const std::shared_ptr<forceField::ForceField> forceField
)
{
    _forceField = forceField;
}

/**
 * @brief set the physical data
 *
 * @param potential
 */
void Evaluator::setPhysicalData(
    const std::shared_ptr<physicalData::PhysicalData> physicalData
)
{
    _physicalData = physicalData;
}

/**
 * @brief set the old physical data
 *
 * @param potential
 */
void Evaluator::setPhysicalDataOld(
    const std::shared_ptr<physicalData::PhysicalData> physicalDataOld
)
{
    _physicalDataOld = physicalDataOld;
}

/**
 * @brief set the virial
 *
 * @param virial
 */
void Evaluator::setVirial(const std::shared_ptr<virial::Virial> virial)
{
    _virial = virial;
}

/**
 * @brief set the intra non bonded
 *
 * @param i
 */
void Evaluator::setIntraNonBonded(
    const std::shared_ptr<intraNonBonded::IntraNonBonded> intraNonBonded
)
{
    _intraNonBonded = intraNonBonded;
}