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

#include "intraNonBondedSetup.hpp"

#include "engine.hpp"           // for Engine
#include "intraNonBonded.hpp"   // for IntraNonBonded
#include "potential.hpp"        // for Potential

using namespace setup;
using namespace engine;

/**
 * @brief wrapper to construct IntraNonBondedSetup object and setup the intra
 * non bonded interactions
 *
 * @param engine
 */
void setup::setupIntraNonBonded(engine::Engine &engine)
{
    if (!engine.isIntraNonBondedActivated())
        return;

    engine.getStdoutOutput().writeSetup("Intra Non-Bonded Interactions");
    engine.getLogOutput().writeSetup("Intra Non-Bonded Interactions");

    IntraNonBondedSetup intraNonBondedSetup(engine);
    intraNonBondedSetup.setup();
}

/**
 * @brief Construct a new Intra Non Bonded Setup:: Intra Non Bonded Setup object
 *
 * @param engine
 */
IntraNonBondedSetup::IntraNonBondedSetup(Engine &engine) : _engine(engine){};

/**
 * @brief Setup intra non bonded interactions
 *
 * @details Setup the coulombPotential and nonCoulombPotential in the
 * IntraNonBonded class. Then the IntraNonBonded maps vector is filled with all
 * single intraNonBonded maps. A single intraNonBonded map is a class containing
 * the molecule pointer and the IntraNonBonded container which represents the
 * molecule type of the molecule pointer.
 *
 */
void IntraNonBondedSetup::setup()
{
    auto       &intraNonBonded = _engine.getIntraNonBonded();
    const auto &potential      = _engine.getPotential();
    const auto &nonCoulombPot  = potential.getNonCoulombPotSharedPtr();
    const auto &coulombPot     = potential.getCoulombPotSharedPtr();

    intraNonBonded.setNonCoulombPotential(nonCoulombPot);
    intraNonBonded.setCoulombPotential(coulombPot);

    intraNonBonded.fillIntraNonBondedMaps(_engine.getSimulationBox());
}