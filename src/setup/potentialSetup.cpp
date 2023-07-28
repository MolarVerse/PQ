#include "potentialSetup.hpp"

using namespace std;
using namespace setup;
using namespace potential;

/**
 * @brief wrapper for setup potential
 *
 */
void setup::setupPotential(engine::Engine &engine)
{
    PotentialSetup potentialSetup(engine);
    potentialSetup.setup();
}

/**
 * @brief sets all nonbonded potential types
 *
 */
void PotentialSetup::setup()
{
    if (_engine._potential->getCoulombType() == "guff") _engine._potential->setCoulombPotential(GuffCoulomb());

    if (_engine._potential->getNonCoulombType() == "guff") _engine._potential->setNonCoulombPotential(GuffNonCoulomb());
}
