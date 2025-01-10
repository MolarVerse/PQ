/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <cassert>

#include "engine.hpp"
#include "forceFieldSettings.hpp"
#include "lennardJones.hpp"
#include "potential.hpp"
#include "potentialSettings.hpp"
#include "setup.hpp"

using namespace engine;
using namespace settings;
using namespace potential;

/**
 * @brief setup the engine with the flattened data
 *
 * @param inputFileName
 * @param engine
 */
void setup::setupFlattenedData(Engine &engine)
{
    // TODO: clean the following lines up!

    auto &simBox = engine.getSimulationBox();

    simBox.flattenAtomTypes();
    simBox.flattenMolTypes();
    simBox.flattenInternalGlobalVDWTypes();

#ifdef __PQ_GPU__
    if (engine.getDevice().isDeviceUsed())
        initDeviceMemory(engine);
#endif

    auto *const pot = engine.getPotentialPtr();

    using enum NonCoulombType;

    if (ForceFieldSettings::isActive())
    {
        if (PotentialSettings::getNonCoulombType() == LJ)
            setupFlattenedNonCoulPotFF<LennardJones>(pot);
        else if (PotentialSettings::getNonCoulombType() == BUCKINGHAM)
            setupFlattenedNonCoulPotFF<Buckingham>(pot);
        else if (PotentialSettings::getNonCoulombType() == MORSE)
            setupFlattenedNonCoulPotFF<Morse>(pot);
        else
            customException::UserInputException(
                "NonCoulombType not implemented yet"
            );
    }
    else
    {
        if (PotentialSettings::getNonCoulombType() == LJ)
            setupFlattenedNonCoulPotGuff<LennardJones>(pot, simBox);
        else if (PotentialSettings::getNonCoulombType() == BUCKINGHAM)
            setupFlattenedNonCoulPotGuff<Buckingham>(pot, simBox);
        else if (PotentialSettings::getNonCoulombType() == MORSE)
            setupFlattenedNonCoulPotGuff<Morse>(pot, simBox);
        else
            customException::UserInputException(
                "NonCoulombType only implemented for ForceField at the moment"
            );
    }

    pot->setCoulombParamVectors(pot->getCoulombPotential().copyParamsVector());

    const auto box            = simBox.getBoxPtr();
    const auto isOrthoRhombic = box->isOrthoRhombic();

    pot->setFunctionPointers(isOrthoRhombic);

    pot->initDeviceMemory(engine.getDevice());
    pot->copyNonCoulParamsTo(engine.getDevice());
    pot->copyNonCoulCutOffsTo(engine.getDevice());
    pot->copyCoulParamsTo(engine.getDevice());
}

#ifdef __PQ_GPU__
/**
 * @brief Initialize device memory for simulation box
 *
 */
void setup::initDeviceMemory(engine::Engine &engine)   // TODO: rename this part
{
    auto &simBox = engine.getSimulationBox();

    simBox.copyAtomTypesTo();
    simBox.copyMolTypesTo();
    simBox.copyInternalGlobalVDWTypesTo();
}
#endif