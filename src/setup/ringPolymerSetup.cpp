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
#include <functional>    // for identity
#include <iostream>      // for operator<<, endl, basic_ostream, cout
#include <stddef.h>      // for size_t
#include <string_view>   // for string_view

using setup::RingPolymerSetup;

/**
 * @brief wrapper to build RingPolymerSetup object and call setup
 *
 * @param engine
 */
void setup::setupRingPolymer(engine::RingPolymerEngine &engine)
{
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

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
    {
        simulationBox::SimulationBox bead;
        bead.copy(_engine.getSimulationBox());

        _engine.addRingPolymerBead(bead);
    }

    if (settings::FileSettings::isRingPolymerStartFileNameSet())
    {
        std::cout << "read ring polymer restart file" << std::endl;
        input::ringPolymer::readRingPolymerRestartFile(_engine);
    }
    else
    {
        auto initVelocities = [](auto &bead)
        {
            maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
            maxwellBoltzmann.initializeVelocities(bead);
        };

        std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
    }
}