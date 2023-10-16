/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "ringPolymerSetup.hpp"

#include "exceptions.hpp"                     // for InputFileException
#include "fileSettings.hpp"                   // for FileSettings
#include "maxwellBoltzmann.hpp"               // for MaxwellBoltzmann
#include "ringPolymerEngine.hpp"              // for RingPolymerEngine
#include "ringPolymerRestartFileReader.hpp"   // for readRingPolymerRestartFile
#include "ringPolymerSettings.hpp"            // for RingPolymerSettings
#include "simulationBox.hpp"                  // for SimulationBox

#include <algorithm>     // for __for_each_fn, for_each
#include <cstddef>       // for size_t
#include <functional>    // for identity
#include <iostream>      // for operator<<, endl, basic_ostream, cout
#include <string_view>   // for string_view

#ifdef WITH_MPI
#include "mpi.hpp"   // for MPI

#include <mpi.h>   // for MPI_Bcast, MPI_DOUBLE, MPI_COMM_WORLD
#endif

using setup::RingPolymerSetup;

/**
 * @brief wrapper to build RingPolymerSetup object and call setup
 *
 * @param engine
 */
void setup::setupRingPolymer(engine::RingPolymerEngine &engine)
{
    engine.getStdoutOutput().writeSetup("ring polymer MD (RPMD)");
    engine.getLogOutput().writeSetup("ring polymer MD (RPMD)");

    RingPolymerSetup ringPolymerSetup(engine);
    ringPolymerSetup.setup();
}

/**
 * @brief setup a ring polymer simulation
 *
 */
void RingPolymerSetup::setup()
{
    if (!settings::RingPolymerSettings::isNumberOfBeadsSet())
        throw customException::InputFileException("Number of beads not set for ring polymer simulation");

    setupPhysicalData();

    setupSimulationBox();

    initializeBeads();
}

/**
 * @brief setup physical data for ring polymer simulation
 *
 */
void RingPolymerSetup::setupPhysicalData()
{
    _engine.getPhysicalData().resizeRingPolymerEnergy(settings::RingPolymerSettings::getNumberOfBeads());
    _engine.getAveragePhysicalData().resizeRingPolymerEnergy(settings::RingPolymerSettings::getNumberOfBeads());
}

/**
 * @brief setup simulation box for ring polymer simulation
 *
 */
void RingPolymerSetup::setupSimulationBox()
{
    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
    {
        simulationBox::SimulationBox bead;
        bead.copy(_engine.getSimulationBox());

        _engine.addRingPolymerBead(bead);
    }
}

/**
 * @brief initialize beads for ring polymer simulation
 *
 * @details if no restart file is given, the velocities of the beads are initialized with maxwell boltzmann distribution
 *
 */
void RingPolymerSetup::initializeBeads()
{
    if (settings::FileSettings::isRingPolymerStartFileNameSet())
    {
        std::cout << "read ring polymer restart file" << std::endl;
        input::ringPolymer::readRingPolymerRestartFile(_engine);
    }
    else
        initializeVelocitiesOfBeads();
}

#ifdef WITH_MPI
void RingPolymerSetup::initializeVelocitiesOfBeads()
{
    auto initVelocities = [](auto &bead)
    {
        if (mpi::MPI::isRoot())
        {
            maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
            maxwellBoltzmann.initializeVelocities(bead);
        }

        auto velocities = bead.flattenVelocities();

        ::MPI_Bcast(velocities.data(), velocities.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        bead.deFlattenVelocities(velocities);
    };

    std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
}
#else
void RingPolymerSetup::initializeVelocitiesOfBeads()
{
    auto initVelocities = [](auto &bead)
    {
        maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
        maxwellBoltzmann.initializeVelocities(bead);
    };

    std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
}
#endif