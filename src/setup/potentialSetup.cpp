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
 * @brief sets all nonBonded potential types
 *
 */
void PotentialSetup::setup()
{
    if (_engine._potential->getCoulombType() == "guff")
    {
        if (_engine.getSettings().getCoulombLongRangeType() == "none")
            _engine._potential->setCoulombPotential(GuffCoulomb());
        else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
        {
            auto wolfParameter = _engine.getSettings().getWolfParameter();
            _engine._potential->setCoulombPotential(GuffWolfCoulomb(wolfParameter));
        }
    }

    if (_engine._potential->getNonCoulombType() == "guff") _engine._potential->setNonCoulombPotential(GuffNonCoulomb());
}
