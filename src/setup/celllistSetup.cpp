#include "celllistSetup.hpp"

#include "angleForceField.hpp"   // for potential
#include "celllist.hpp"          // for CellList
#include "engine.hpp"            // for Engine
#include "potential.hpp"         // for PotentialBruteForce, PotentialCellList

using namespace setup;

/**
 * @brief wrapper to build SetupCellList object and call setup
 *
 * @param engine
 */
void setup::setupCellList(engine::Engine &engine)
{
    CellListSetup cellListSetup(engine);
    cellListSetup.setup();
}

/**
 * @brief setup cell list
 *
 * @details if cell list is activated, resize cells, setup cell list and set potential to cell list potential,
 * otherwise set potential to brute force potential. The nonCoulombPotential is stored and set again to new potential object.
 *
 */
void CellListSetup::setup()
{
    auto nonCoulombPotential = _engine.getPotential().getNonCoulombPotentialSharedPtr();

    if (_engine.isCellListActivated())
    {
        _engine.getCellList().resizeCells();
        _engine.getCellList().setup(_engine.getSimulationBox());
        _engine.makePotential(potential::PotentialCellList());
    }
    else
        _engine.makePotential(potential::PotentialBruteForce());

    _engine.getPotential().setNonCoulombPotential(nonCoulombPotential);
}