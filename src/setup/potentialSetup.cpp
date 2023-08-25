#include "potentialSetup.hpp"

#include "angleForceField.hpp"           // for potential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for ParameterFileException
#include "forceField.hpp"                // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "nonCoulombPair.hpp"            // for NonCoulombPair
#include "nonCoulombPotential.hpp"       // for NonCoulombPotential
#include "potential.hpp"                 // for Potential
#include "settings.hpp"                  // for Settings
#include "simulationBox.hpp"             // for SimulationBox

#include <algorithm>     // for __for_each_fn, __sort_fn
#include <cstddef>       // for size_t
#include <functional>    // for identity
#include <memory>        // for swap, shared_ptr, __shared_ptr_access
#include <string>        // for operator==
#include <string_view>   // for string_view
#include <vector>        // for vector

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
}

/**
 * @brief sets nonCoulomb potential type
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
    {
        // _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb()); TODO: think of a clever way to do
        // this
    }
    else
        _engine.getPotential().makeNonCoulombPotential(potential::GuffNonCoulomb());
}

/**
 * @brief TODO: explain in detail!!!!!!!!!!!!!!!!!!!
 *
 */
void PotentialSetup::setupNonCoulombicPairs()
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    std::ranges::for_each(potential.getNonCoulombicPairsVector(),
                          [](auto &nonCoulombicPair)
                          {
                              const auto &[energy, force] =
                                  nonCoulombicPair->calculateEnergyAndForce(nonCoulombicPair->getRadialCutOff());
                              nonCoulombicPair->setEnergyCutOff(energy);
                              nonCoulombicPair->setForceCutOff(force);
                          });

    _engine.getSimulationBox().setupExternalToInternalGlobalVdwTypesMap();
    potential.determineInternalGlobalVdwTypes(_engine.getSimulationBox().getExternalToInternalGlobalVDWTypes());

    const auto numberOfGlobalVdwTypes           = _engine.getSimulationBox().getExternalGlobalVdwTypes().size();
    auto       selfInteractionNonCoulombicPairs = potential.getSelfInteractionNonCoulombicPairs();

    if (selfInteractionNonCoulombicPairs.size() != numberOfGlobalVdwTypes)
        throw customException::ParameterFileException(
            "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    std::ranges::sort(selfInteractionNonCoulombicPairs,
                      [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
                      { return nonCoulombicPair1->getInternalType1() < nonCoulombicPair2->getInternalType1(); });

    for (size_t i = 0; i < numberOfGlobalVdwTypes; ++i)
        if (selfInteractionNonCoulombicPairs[i]->getInternalType1() != i)
            throw customException::ParameterFileException(
                "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    potential.fillDiagonalElementsOfNonCoulombicPairsMatrix(selfInteractionNonCoulombicPairs);
    potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();
}