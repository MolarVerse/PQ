#include "intraNonBondedSetup.hpp"

using namespace setup;

/**
 * @brief Setup intra non bonded interactions
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