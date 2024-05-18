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
#include "nonCoulombPotential.hpp"
#include "potentialSettings.hpp"
#include "settings.hpp"
#include "simulationBox_kokkos.hpp"

using namespace setup;

/**
 * @brief setup Kokkos
 */
void setup::setupKokkos(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Kokkos");
    engine.getLogOutput().writeSetup("Kokkos");

    KokkosSetup kokkosSetup(engine);
    kokkosSetup.setup();
}

/**
 * @brief setup Kokkos
 */
void KokkosSetup::setup()
{
    if (!settings::Settings::isMMActivated())
        return;

    if (settings::PotentialSettings::getNonCoulombType() !=
        settings::NonCoulombType::LJ)
    {
        auto warning = customException::UserInputExceptionWarning(
            "Kokkos installation is not enabled for the current type of non "
            "Coulomb potential - falling back to serial execution"
        );
        std::cerr << warning.what() << std::endl;
        return;
    }

    if (settings::PotentialSettings::getCoulombLongRangeType() != "wolf")
    {
        auto warning = customException::UserInputExceptionWarning(
            "Kokkos installation is not enabled for the current type of "
            "Coulomb long range potential - falling back to serial execution"
        );
        std::cerr << warning.what() << std::endl;
        return;
    }

    settings::Settings::activateKokkos();

    _engine.initKokkosPotential();

    const auto numAtoms = _engine.getSimulationBox().getNumberOfAtoms();

    _engine.initKokkosSimulationBox(numAtoms);

    auto kokkosSimulationBox = _engine.getKokkosSimulationBox();

    kokkosSimulationBox.transferAtomTypesFromSimulationBox(
        _engine.getSimulationBox()
    );
    kokkosSimulationBox.transferMolTypesFromSimulationBox(
        _engine.getSimulationBox()
    );
    kokkosSimulationBox.transferInternalGlobalVDWTypesFromSimulationBox(
        _engine.getSimulationBox()
    );
    kokkosSimulationBox.transferPartialChargesFromSimulationBox(
        _engine.getSimulationBox()
    );

    const auto numAtomTypes =
        _engine.getSimulationBox().getExternalGlobalVdwTypes().size();

    _engine.initKokkosLennardJones(numAtomTypes);

    auto kokkosLennardJones = _engine.getKokkosLennardJones();

    auto forceFieldNonCoulomb = dynamic_cast<potential::ForceFieldNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    kokkosLennardJones.transferFromNonCoulombPairMatrix(
        forceFieldNonCoulomb.getNonCoulombPairsMatrix()
    );

    auto wolfPotential = dynamic_cast<potential::CoulombWolf &>(
        _engine.getPotential().getCoulombPotential()
    );

    _engine.initKokkosCoulombWolf(
        wolfPotential.getCoulombRadiusCutOff(),
        wolfPotential.getKappa(),
        wolfPotential.getWolfParameter1(),
        wolfPotential.getWolfParameter2(),
        wolfPotential.getWolfParameter3(),
        constants::_COULOMB_PREFACTOR_
    );
}