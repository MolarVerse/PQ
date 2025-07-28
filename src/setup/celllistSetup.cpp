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

#include "celllistSetup.hpp"

#include "celllist.hpp"   // for CellList
#include "engine.hpp"     // for Engine
#include "potential.hpp"   // for PotentialBruteForce, PotentialCellList, Potential

using namespace setup;
using namespace engine;
using namespace potential;

/**
 * @brief wrapper to build SetupCellList object and call setup
 *
 * @param engine
 */
void setup::setupCellList(Engine &engine)
{
    if (engine.isCellListActivated())
    {
        engine.getStdoutOutput().writeSetup("Cell List");
        engine.getLogOutput().writeSetup("Cell List");
    }

    CellListSetup cellListSetup(engine);
    cellListSetup.setup();
}

/**
 * @brief constructor
 *
 * @param engine
 */
CellListSetup::CellListSetup(Engine &engine) : _engine(engine) {}

/**
 * @brief setup cell list
 *
 * @details if cell list is activated, resize cells, setup cell list and set
 * potential to cell list potential, otherwise set potential to brute force
 * potential. The nonCoulombPotential is stored and set again to new potential
 * object.
 *
 */
void CellListSetup::setup()
{
    const auto &potential     = _engine.getPotential();
    const auto  nonCoulombPot = potential.getNonCoulombPotSharedPtr();

    if (_engine.isCellListActivated())
    {
        _engine.getCellList().resizeCells();
        _engine.getCellList().setup(_engine.getSimulationBox());
        _engine.makePotential(PotentialCellList());
    }
    else
        _engine.makePotential(PotentialBruteForce());

    _engine.getPotential().setNonCoulombPotential(nonCoulombPot);
}