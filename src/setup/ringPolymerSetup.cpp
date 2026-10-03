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

#include <algorithm>
#include <cstddef>

#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "maxwellBoltzmann.hpp"
#include "ringPolymerEngine.hpp"
#include "ringPolymerRestartFileReader.hpp"
#include "ringPolymerSettings.hpp"
#include "simulationBox.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

namespace setup
{

    /**
     * @brief wrapper to build RingPolymerSetup object and call setup
     *
     * @param engine
     */
    void setupRingPolymer(engine::Engine &engine)
    {
        if (!settings::GeneralSettings::isRingPolymerMDActivated())
        {
#ifdef WITH_MPI
            if (mpi::MPI::getSize() > 1)
                throw exc::MPIException(
                    "MPI parallelization with more than one process is not "
                    "supported for non-ring polymer MD"
                );
#endif

            return;
        }

        out::StdoutOutput::writeSetup("Ring Polymer MD (RPMD)");
        engine.getLogOutput().writeSetup("Ring Polymer MD (RPMD)");

        RingPolymerSetup ringPolySetup(
            dynamic_cast<engine::RingPolymerEngine &>(engine)
        );
        ringPolySetup.setup();
    }

    /**
     * @brief Construct a new Ring Polymer Setup object
     *
     * @param engine
     */
    RingPolymerSetup::RingPolymerSetup(engine::RingPolymerEngine &engine)
        : _engine(engine)
    {
    }

    /**
     * @brief setup a ring polymer simulation
     *
     */
    void RingPolymerSetup::setup()
    {
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
        const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();
        _engine.resizeRingPolymerBeadPhysicalData(nBeads);
    }

    /**
     * @brief setup simulation box for ring polymer simulation
     *
     */
    void RingPolymerSetup::setupSimulationBox()
    {
        for (size_t i = 0;
             i < settings::RingPolymerSettings::getNumberOfBeads();
             ++i)
        {
            molsys::SimulationBox bead;
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
        if (settings::FileSettings::isRingPolymerStartFileNameSet())
        {
            auto             &log = _engine.getLogOutput();
            const auto *const msg = "Reading ring polymer restart file: ";
            const auto       &file =
                settings::FileSettings::getRingPolymerStartFileName();

            log.writeRead(msg, file);
            out::StdoutOutput::writeRead(msg, file);

            input::ringPolymer::readRingPolymerRestartFile(_engine);
        }
        else
        {
            initializeVelocitiesOfBeads();
        }
    }

    /**
     * @brief initialize velocities of beads with maxwell boltzmann distribution
     *
     */
    void RingPolymerSetup::initializeVelocitiesOfBeads()
    {
        auto initVelocities = [](auto &bead)
        {
            maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
            maxwellBoltzmann.initializeVelocities(bead);
        };

        std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
    }

}   // namespace setup
