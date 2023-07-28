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
void CellListSetup::setup()
{
    if (_engine.getCellList().isActivated())
    {
        _engine.getCellList().setup(_engine.getSimulationBox());
        _engine._potential = make_unique<PotentialCellList>();
    }
    else
    {
        _engine._potential = make_unique<PotentialBruteForce>();
    }
}