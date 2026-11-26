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

#include "celllist.hpp"
#include "engine.hpp"
#include "generalSettings.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"

namespace setup
{

    /**
     * @brief wrapper to build SetupCellList object and call setup
     *
     * @param engine
     */
    void setupCellList(engine::Engine &engine)
    {
        if (engine.getSettings().generalSettings.isCellListActivated())
        {
            out::StdoutOutput::writeSetup("Cell List");
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
    CellListSetup::CellListSetup(engine::Engine &engine) : _engine(engine) {}

    /**
     * @brief setup cell list
     *
     * @details if cell list is activated, resize cells, setup cell list and set
     * potential to cell list potential, otherwise set potential to brute force
     * potential. The nonCoulombPotential is stored and set again to new
     * potential object.
     *
     */
    void CellListSetup::setup()
    {
        const auto &potential     = _engine.getPotential();
        const auto  nonCoulombPot = potential->getNonCoulombPotSharedPtr();

        if (_engine.getSettings().generalSettings.isCellListActivated())
        {
            auto &cellList = _engine.getCellList();
            cellList.setNumberOfCells(
                _engine.getSettings().generalSettings.getNumberOfCells()
            );
            cellList.resizeCells();
            cellList.setup(
                _engine.get_simulationBox(),
                _engine.getSettings().potentialSettings.getCoulombRadiusCutoff()
            );
            _engine.makePotential(
                pot::PotentialCellList(_engine.getSettings().hybridSettings)
            );
        }
        else
        {
            _engine.makePotential(
                pot::PotentialBruteForce(_engine.getSettings().hybridSettings)
            );
        }

        _engine.getPotential()->setNonCoulombPotential(nonCoulombPot);
    }

}   // namespace setup
