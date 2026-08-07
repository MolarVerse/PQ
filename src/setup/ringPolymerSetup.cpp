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

#include "ringPolymerSetup.hpp"

#include <algorithm>     // for __for_each_fn, for_each
#include <cstddef>       // for size_t
#include <functional>    // for identity
#include <iostream>      // for operator<<, endl, basic_ostream, cout
#include <string_view>   // for string_view

#include "exceptions.hpp"                     // for InputFileException
#include "fileSettings.hpp"                   // for FileSettings
#include "maxwellBoltzmann.hpp"               // for MaxwellBoltzmann
#include "ringPolymerEngine.hpp"              // for RingPolymerEngine
#include "ringPolymerRestartFileReader.hpp"   // for readRingPolymerRestartFile
#include "ringPolymerSettings.hpp"            // for RingPolymerSettings
#include "settings.hpp"                       // for Settings
#include "simulationBox.hpp"                  // for SimulationBox

#ifdef WITH_MPI
#include "mpi.hpp"   // for MPI
#endif

using setup::RingPolymerSetup;
using namespace engine;
using namespace settings;
using namespace customException;
using namespace input::ringPolymer;
using namespace maxwellBoltzmann;

/**
 * @brief wrapper to build RingPolymerSetup object and call setup
 *
 * @param engine
 */
void setup::setupRingPolymer(Engine &engine)
{
    if (!Settings::isRingPolymerMDActivated())
    {
#ifdef WITH_MPI
        if (mpi::MPI::getSize() > 1)
            throw MPIException(
                "MPI parallelization with more than one process is not "
                "supported for non-ring polymer MD"
            );
#endif

        return;
    }

    engine.getStdoutOutput().writeSetup("Ring Polymer MD (RPMD)");
    engine.getLogOutput().writeSetup("Ring Polymer MD (RPMD)");

    RingPolymerSetup ringPolySetup(dynamic_cast<RingPolymerEngine &>(engine));
    ringPolySetup.setup();
}

/**
 * @brief Construct a new Ring Polymer Setup object
 *
 * @param engine
 */
RingPolymerSetup::RingPolymerSetup(RingPolymerEngine &engine)
    : _engine(engine){};

/**
 * @brief setup a ring polymer simulation
 *
 */
void RingPolymerSetup::setup()
{
    if (!RingPolymerSettings::isNumberOfBeadsSet())
        throw InputFileException(
            "Number of beads not set for ring polymer simulation"
        );

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
    const auto nBeads = RingPolymerSettings::getNumberOfBeads();
    _engine.resizeRingPolymerBeadPhysicalData(nBeads);
}

/**
 * @brief setup simulation box for ring polymer simulation
 *
 */
void RingPolymerSetup::setupSimulationBox()
{
    for (size_t i = 0; i < RingPolymerSettings::getNumberOfBeads(); ++i)
    {
        simulationBox::SimulationBox bead;
        bead.copy(_engine.getSimulationBox());

        _engine.addRingPolymerBead(bead);
    }
}

/**
 * @brief initialize beads for ring polymer simulation
 *
 * @details if no restart file is given, the velocities of the beads are
 * initialized with maxwell boltzmann distribution
 *
 */
void RingPolymerSetup::initializeBeads()
{
    if (FileSettings::isRingPolymerStartFileNameSet())
    {
        auto       &log    = _engine.getLogOutput();
        const auto &stdOut = _engine.getStdoutOutput();
        const auto &msg    = "Reading ring polymer restart file: ";
        const auto &file   = FileSettings::getRingPolymerStartFileName();

        log.writeRead(msg, file);
        stdOut.writeRead(msg, file);

        readRingPolymerRestartFile(_engine);
    }
    else
        initializeVelocitiesOfBeads();
}

/**
 * @brief initialize velocities of beads with maxwell boltzmann distribution
 *
 */
void RingPolymerSetup::initializeVelocitiesOfBeads()
{
    auto initVelocities = [](auto &bead)
    {
        MaxwellBoltzmann maxwellBoltzmann;
        maxwellBoltzmann.initializeVelocities(bead);
    };

    std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
}