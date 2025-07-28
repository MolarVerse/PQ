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

#include "kokkosSetup.hpp"

#include <iostream>

#include "constants/conversionFactors.hpp"
#include "coulombWolf.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "mdEngine.hpp"
#include "nonCoulombPotential.hpp"
#include "potentialSettings.hpp"
#include "settings.hpp"
#include "simulationBox_kokkos.hpp"
#include "timingsSettings.hpp"
#include "typeAliases.hpp"

using namespace setup;
using namespace engine;
using namespace settings;
using namespace customException;
using namespace potential;
using namespace constants;

/**
 * @brief setup Kokkos
 */
void setup::setupKokkos(Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Kokkos");
    engine.getLogOutput().writeSetup("Kokkos");

    KokkosSetup kokkosSetup(engine);
    kokkosSetup.setup();
}

/**
 * @brief Construct a new Kokkos Setup:: Kokkos Setup object
 *
 * @param engine
 */
KokkosSetup::KokkosSetup(Engine &engine) : _engine(engine) {}

/**
 * @brief setup Kokkos
 */
void KokkosSetup::setup()
{
    if (!Settings::isMMActivated())
        return;

    if (PotentialSettings::getNonCoulombType() != NonCoulombType::LJ)
    {
        const auto warning = UserInputExceptionWarning(
            "Kokkos installation is not enabled for the current type of non "
            "Coulomb potential - falling back to serial execution"
        );
        std::cerr << warning.what() << std::endl;
        return;
    }

    using enum CoulombLongRangeType;
    if (PotentialSettings::getCoulombLongRangeType() != WOLF)
    {
        const auto warning = UserInputExceptionWarning(
            "Kokkos installation is not enabled for the current type of "
            "Coulomb long range potential - falling back to serial execution"
        );
        std::cerr << warning.what() << std::endl;
        return;
    }

    Settings::activateKokkos();

    _engine.initKokkosPotential();

    auto       &simBox        = _engine.getSimulationBox();
    const auto &potential     = _engine.getPotential();
    const auto &nonCoulombPot = potential.getNonCoulombPotential();
    const auto &coulombPot    = potential.getCoulombPotential();

    const auto numAtoms = simBox.getNumberOfAtoms();

    /************************************
     * Initialize Kokkos simulation box *
     ************************************/

    _engine.initKokkosSimulationBox(numAtoms);
    auto kokkosSimulationBox = _engine.getKokkosSimulationBox();
    kokkosSimulationBox.initKokkosSimulationBox(simBox);

    auto ffNonCoulomb = dynamic_cast<const pq::FFNonCoulomb &>(nonCoulombPot);

    /************************************
     * Initialize Kokkos Lennard Jones  *
     ************************************/

    const auto numAtomTypes = ffNonCoulomb.getNonCoulombPairsMatrix().rows();
    _engine.initKokkosLennardJones(numAtomTypes);
    auto kokkosLennardJones = _engine.getKokkosLennardJones();

    auto &nonCoulPairMatrix = ffNonCoulomb.getNonCoulombPairsMatrix();
    kokkosLennardJones.transferFromNonCoulombPairMatrix(nonCoulPairMatrix);

    /************************************
     * Initialize Kokkos Coulomb Wolf   *
     ************************************/

    const auto wolfPotential = dynamic_cast<const CoulombWolf &>(coulombPot);

    _engine.initKokkosCoulombWolf(
        CoulombWolf::getCoulombRadiusCutOff(),
        wolfPotential.getKappa(),
        wolfPotential.getWolfParameter1(),
        wolfPotential.getWolfParameter2(),
        wolfPotential.getWolfParameter3(),
        _COULOMB_PREFACTOR_
    );
}