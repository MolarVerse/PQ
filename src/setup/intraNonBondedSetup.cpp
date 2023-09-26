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

#include "intraNonBondedSetup.hpp"

#include "engine.hpp"           // for Engine
#include "intraNonBonded.hpp"   // for IntraNonBonded
#include "potential.hpp"        // for Potential

using namespace setup;

/**
 * @brief Setup intra non bonded interactions
 *
 * @details Setup the coulombPotential and nonCoulombPotential in the IntraNonBonded class. Then the IntraNonBonded maps vector is
 * filled with all single intraNonBonded maps. A single intraNonBonded map is a class containing the molecule pointer and the
 * IntraNonBonded container which represents the molecule type of the molecule pointer.
 *
 */
void IntraNonBondedSetup::setup()
{
    if (!_engine.isIntraNonBondedActivated())
        return;

    _engine.getIntraNonBonded().setNonCoulombPotential(_engine.getPotential().getNonCoulombPotentialSharedPtr());
    _engine.getIntraNonBonded().setCoulombPotential(_engine.getPotential().getCoulombPotentialSharedPtr());

    _engine.getIntraNonBonded().fillIntraNonBondedMaps(_engine.getSimulationBox());
}

/**
 * @brief wrapper to construct IntraNonBondedSetup object and setup the intra non bonded interactions
 *
 * @param engine
 */
void setup::setupIntraNonBonded(engine::Engine &engine)
{
    IntraNonBondedSetup intraNonBondedSetup(engine);
    intraNonBondedSetup.setup();
}