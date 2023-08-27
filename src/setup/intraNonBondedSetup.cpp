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