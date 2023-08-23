#include "celllistSetup.hpp"

using namespace std;
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
void CellListSetup::setup()   // TODO: define here a copy constructor otherwise this does not work for noncoulombic pairs which
                              // are set while readine parameter file
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