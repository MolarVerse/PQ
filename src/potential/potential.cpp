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

#include "potential.hpp"

#include "box.hpp"                   // for Box
#include "coulombPotential.hpp"      // for CoulombPotential
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

void Potential::calculateQMMMForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    calculateForces(simBox, physicalData, cellList);
    calculateCoreToOuterForces(simBox, physicalData, cellList);
    calculateLayerToOuterForces(simBox, physicalData, cellList);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the coulomb potential as a shared pointer
 *
 * @param pot
 */
void Potential::setNonCoulombPotential(
    const std::shared_ptr<NonCoulombPotential> pot
)
{
    _nonCoulombPot = pot;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief get the coulomb potential
 *
 * @return CoulombPotential&
 */
CoulombPotential &Potential::getCoulombPotential() const
{
    return *_coulombPotential;
}

/**
 * @brief get the non-coulomb potential
 *
 * @return NonCoulombPotential&
 */
NonCoulombPotential &Potential::getNonCoulombPotential() const
{
    return *_nonCoulombPot;
}

/**
 * @brief get the coulomb potential as a shared pointer
 *
 * @return SharedCoulombPot
 */
std::shared_ptr<CoulombPotential> Potential::getCoulombPotSharedPtr() const
{
    return _coulombPotential;
}

/**
 * @brief get the non-coulomb potential as a shared pointer
 *
 * @return SharedNonCoulombPot
 */
std::shared_ptr<NonCoulombPotential> Potential::getNonCoulombPotSharedPtr(
) const
{
    return _nonCoulombPot;
}