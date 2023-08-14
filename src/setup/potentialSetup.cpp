#include "potentialSetup.hpp"

using namespace std;
using namespace setup;
using namespace potential;
using namespace potential_new;   // TODO:

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
    auto       wolfParameter       = _engine.getSettings().getWolfParameter();

    if (_engine.getPotential().getCoulombType() == "guff")
    {
        if (_engine.getSettings().getCoulombLongRangeType() == "none")
            _engine.getPotential().setCoulombPotential(GuffCoulomb(coulombRadiusCutOff));
        else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
        {
            _engine.getPotential().setCoulombPotential(GuffWolfCoulomb(coulombRadiusCutOff, wolfParameter));
        }
    }

    // TODO: from here on only potential new

    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
    {
        if (_engine.getSettings().getCoulombLongRangeType() == "none")
            _engine.getPotentialNew().makeCoulombPotential(ForceFieldShiftedPotential(coulombRadiusCutOff));
        else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
        {
            _engine.getPotentialNew().makeCoulombPotential(ForceFieldWolf(coulombRadiusCutOff, wolfParameter));
        }
    }
    else
    {
        if (_engine.getSettings().getCoulombLongRangeType() == "none")
            _engine.getPotentialNew().makeCoulombPotential(GuffCoulombShiftedPotential(coulombRadiusCutOff));
        else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
        {
            _engine.getPotentialNew().makeCoulombPotential(GuffCoulombWolf(coulombRadiusCutOff, wolfParameter));
        }
    }
}

/**
 * @brief sets nonCoulomb potential type
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    if (_engine.getPotential().getNonCoulombType() == "guff")
    {
        if (_engine.getSettings().getNonCoulombType() == "none")
            _engine.getPotential().setNonCoulombPotential(potential::GuffNonCoulomb());
        else if (_engine.getSettings().getNonCoulombType() == "lj")
            _engine.getPotential().setNonCoulombPotential(GuffLennardJones());
        else if (_engine.getSettings().getNonCoulombType() == "buck")
            _engine.getPotential().setNonCoulombPotential(GuffBuckingham());
    }

    // TODO: from here on only potential new

    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        _engine.getPotentialNew().makeNonCoulombPotential(potential_new::ForceFieldNonCoulomb());
    else
        _engine.getPotentialNew().makeNonCoulombPotential(potential_new::GuffNonCoulomb());
}