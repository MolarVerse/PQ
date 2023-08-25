#include "celllistSetup.hpp"

#include "angleForceField.hpp"   // for potential
#include "celllist.hpp"          // for CellList
#include "engine.hpp"            // for Engine
#include "potential.hpp"         // for PotentialBruteForce, PotentialCellList

using namespace setup;
using namespace potential;

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
 */
void CellListSetup::setup()
{
    auto nonCoulombPotential = _engine.getPotential().getNonCoulombPotentialSharedPtr();
    if (_engine.isCellListActivated())
    {
        _engine.getCellList().setup(_engine.getSimulationBox());
        _engine.makePotential(PotentialCellList());
    }
    else
    {
        _engine.makePotential(PotentialBruteForce());
    }
    _engine.getPotential().setNonCoulombPotential(nonCoulombPotential);
}