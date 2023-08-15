#include "potentialSetup.hpp"

#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "guffNonCoulomb.hpp"

#include <ranges>

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

    if (!_engine.isForceFieldNonCoulombicsActivated())
        return;

    setupNonCoulombicPairs();
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

    if (_engine.getSettings().getCoulombLongRangeType() == "none")
        _engine.getPotential().makeCoulombPotential(CoulombShiftedPotential(coulombRadiusCutOff));
    else if (_engine.getSettings().getCoulombLongRangeType() == "wolf")
    {
        _engine.getPotential().makeCoulombPotential(CoulombWolf(coulombRadiusCutOff, wolfParameter));
    }

    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        _engine.getPotential().getCoulombPotential().setCoulombPreFactorToForceField();
    else
        _engine.getPotential().getCoulombPotential().setCoulombPreFactorToGuff();
}

/**
 * @brief sets nonCoulomb potential type
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    else
        _engine.getPotential().makeNonCoulombPotential(potential::GuffNonCoulomb());
}

void PotentialSetup::setupNonCoulombicPairs()
{
    ranges::for_each(_engine.getPotential().getNonCoulombicPairsVector(),
                     [](auto &nonCoulombicPair)
                     {
                         const auto &[energy, force] =
                             nonCoulombicPair->calculateEnergyAndForce(nonCoulombicPair->getRadialCutOff());
                         nonCoulombicPair->setEnergyCutOff(energy);
                         nonCoulombicPair->setForceCutOff(force);
                     });

    _engine.getSimulationBox().setupExternalToInternalGlobalVdwTypesMap();

    // _engine.getForceFieldPtr()->determineInternalGlobalVdwTypes(_engine.getSimulationBox().getExternalToInternalGlobalVDWTypes());
}