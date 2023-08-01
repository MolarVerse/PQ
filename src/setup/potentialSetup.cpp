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
    setupCoulomb();
    setupNonCoulomb();
}

/**
 * @brief sets coulomb potential type
 *
 * @param coulombType
 */
void PotentialSetup::setupCoulomb()
{
    const auto coulombRadiusCutOff = _engine.getSimulationBox().getCoulombRadiusCutOff();

    if (_engine._potential->getCoulombType() == "guff")
    {
        if (_engine.getSettings().getCoulombLongRangeType() == "none")
            _engine._potential->setCoulombPotential(GuffCoulomb(coulombRadiusCutOff));
        else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
        {
            auto wolfParameter = _engine.getSettings().getWolfParameter();
            _engine._potential->setCoulombPotential(GuffWolfCoulomb(coulombRadiusCutOff, wolfParameter));
        }
    }
}

/**
 * @brief sets nonCoulomb potential type
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    if (_engine._potential->getNonCoulombType() == "guff")
    {
        if (_engine.getSettings().getNonCoulombType() == "none")
            _engine._potential->setNonCoulombPotential(GuffNonCoulomb());
        else if (_engine.getSettings().getNonCoulombType() == "lj")
            _engine._potential->setNonCoulombPotential(GuffLennardJones());
        else if (_engine.getSettings().getNonCoulombType() == "buck")
            _engine._potential->setNonCoulombPotential(GuffBuckingham());
    }
}