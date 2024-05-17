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
        customException::UserInputExceptionWarning(
            "Kokkos installation is not enabled for the current type of non "
            "Coulomb potential - falling back to serial execution"
        );
        return;
    }

    settings::Settings::activateKokkos();
    Kokkos::initialize();

    auto numAtoms = _engine.getSimulationBox().getNumberOfAtoms();

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

    auto numAtomTypes =
        _engine.getSimulationBox().getExternalGlobalVdwTypes().size();

    _engine.initKokkosLennardJones(numAtomTypes);

    auto kokkosLennardJones = _engine.getKokkosLennardJones();

    auto forceFieldNonCoulomb = dynamic_cast<potential::ForceFieldNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    kokkosLennardJones.transferFromNonCoulombPairMatrix(
        forceFieldNonCoulomb.getNonCoulombPairsMatrix()
    );
}