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

#include "coulombPotential.hpp"
#include "engine.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "forceFieldSettings.hpp"
#include "lennardJones.hpp"
#include "lennardJonesPair.hpp"
#include "nonCoulombPotential.hpp"
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
    simBox.flattenPositions();
    simBox.flattenVelocities();
    simBox.flattenForces();
    simBox.flattenShiftForces();
    simBox.flattenCharges();

    simBox.initAtomsPerMolecule();
    simBox.initMoleculeIndices();

    simBox.flattenAtomTypes();
    simBox.flattenMolTypes();
    simBox.flattenInternalGlobalVDWTypes();

#ifdef __PQ_GPU__
    if (engine.getDevice().isDeviceUsed())
        initDeviceMemory(engine);
#endif

    auto *const pot = engine.getPotentialPtr();

    if (PotentialSettings::getNonCoulombType() == NonCoulombType::LJ)
    {
        auto &nonCoulombPot = pot->getNonCoulombPotential();

        setupFlattenedLJ(nonCoulombPot, pot);
    }

    pot->setCoulombParamVectors(pot->getCoulombPotential().copyParamsVector());

    const auto box            = simBox.getBoxPtr();
    const auto isOrthoRhombic = box->isOrthoRhombic();

    pot->setFunctionPointers(isOrthoRhombic);
}

/**
 * @brief setup the flattened Lennard-Jones potential
 *
 * @param nonCoulPot
 * @param potential
 */
void setup::setupFlattenedLJ(
    NonCoulombPotential &nonCoulPot,
    Potential *const     potential
)
{
    std::vector<Real> params;
    std::vector<Real> cutOffs;

    size_t size = 0;

    if (ForceFieldSettings::isActive())
    {
        auto &nonCoulombPot = dynamic_cast<ForceFieldNonCoulomb &>(nonCoulPot);

        auto &nonCoulombPairs = nonCoulombPot.getNonCoulombPairsMatrix();
        const auto [row, col] = nonCoulombPairs.shape();

        assert(row == col);

        auto lennardJones = LennardJones();
        lennardJones.resize(row);

        // clang-format off
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < row; ++i)
            for (size_t j = 0; j < col; ++j)
            {
                const auto &pair =
                    dynamic_cast<LennardJonesPair &>(*nonCoulombPairs(i, j));

                lennardJones.addPair(pair, i, j);
            }
        // clang-format on

        params  = lennardJones.copyParams();
        cutOffs = lennardJones.copyCutOffs();
        size    = lennardJones.getSize();
    }
    else
    {
        throw customException::NotImplementedException(
            "The nonCoulomb potential is not implemented yet"
        );
    }

    potential->setNonCoulombParamVectors(
        params,
        cutOffs,
        LennardJones::getNParams(),
        size
    );
}