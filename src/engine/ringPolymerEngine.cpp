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

#include "ringPolymerEngine.hpp"

#include <algorithm>
#include <cstddef>

#include "atom.hpp"
#include "constants/internalConversionFactors.hpp"
#include "engineOutput.hpp"
#include "globalTimer.hpp"
#include "outputFileSettings.hpp"
#include "physicalData.hpp"
#include "ringPolymerSettings.hpp"
#include "thermostatSettings.hpp"
#include "timingsSettings.hpp"

namespace engine
{

    /**
     * @brief resizes the vector of physical data for the ring polymer beads
     *
     * @param numberOfBeads
     */
    void RingPolymerEngine::resizeRingPolymerBeadPhysicalData(
        size_t numberOfBeads
    )
    {
        _ringPolymerBeadsPhysicalData.resize(numberOfBeads);
        _averageRingPolymerBeadsPhysicalData.resize(numberOfBeads);
    }

    /**
     * @brief writes the ring polymer output files.
     *
     */
    void RingPolymerEngine::writeOutput()
    {
        auto &averageRPMDData = _averageRingPolymerBeadsPhysicalData;
        auto &rpmdData        = _ringPolymerBeadsPhysicalData;

        const auto outputFreq =
            settings::OutputFileSettings::getOutputFrequency();
        const auto step0   = settings::TimingsSettings::getStepCount();
        const auto effStep = getStep() + step0;

        if (0 == getStep() % outputFreq)
        {
            getEngineOutput().writeXyzFile(getSimulationBox(), effStep);
            getEngineOutput().writeVelFile(getSimulationBox(), effStep);
            getEngineOutput().writeForceFile(getSimulationBox(), effStep);
            getEngineOutput().writeChargeFile(getSimulationBox(), effStep);
            getEngineOutput()
                .writeRstFile(getSimulationBox(), *_thermostat, effStep);

            getEngineOutput().writeRingPolymerRstFile(_ringPolymerBeads);
            getEngineOutput().writeRingPolymerXyzFile(
                _ringPolymerBeads,
                effStep
            );
            getEngineOutput().writeRingPolymerVelFile(
                _ringPolymerBeads,
                effStep
            );
            getEngineOutput().writeRingPolymerForceFile(
                _ringPolymerBeads,
                effStep
            );
            getEngineOutput().writeRingPolymerChargeFile(
                _ringPolymerBeads,
                effStep
            );
        }

        // NOTE:
        // stop and restart immediately time manager - maximum lost time is en
        // file writing in last step of simulation but on the other hand setup
        // is now included in total simulation time Unfortunately, setup is
        // therefore included in the first looptime output but this is not a big
        // problem - could also be a feature and not a bug
        timings::GlobalTimer::get().stopAndRestartSimulationTimer();
        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime();

        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        {
            rpmdData[i].setLoopTime(elapsedTime);
            averageRPMDData[i].updateAverages(rpmdData[i]);
        }

        if (0 == getStep() % outputFreq)
        {
            for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
                averageRPMDData[i].makeAverages(
                    static_cast<double>(outputFreq)
                );

            getPhysicalData().copy(mean(rpmdData));
            getAveragePhysicalData() = mean(averageRPMDData);

            const auto timeStep      = settings::TimingsSettings::getTimeStep();
            const auto effStepDouble = static_cast<double>(effStep);
            const auto simTime       = effStepDouble * timeStep * FS_TO_PS;

            getEngineOutput().writeEnergyFile(
                effStep,
                getAveragePhysicalData()
            );
            getEngineOutput().writeInstantEnergyFile(
                effStep,
                getPhysicalData()
            );
            getEngineOutput().writeMomentumFile(
                effStep,
                getAveragePhysicalData()
            );
            getEngineOutput().writeInfoFile(simTime, getAveragePhysicalData());

            getEngineOutput().writeRingPolymerEnergyFile(
                step0 + getStep(),
                averageRPMDData
            );

            for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
                averageRPMDData[i] = physicalData::PhysicalData();

            getAveragePhysicalData() = physicalData::PhysicalData();
        }

        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
            rpmdData[i].reset();
    }

    /**
     * @brief coupling step of ring polymers
     *
     */
    void RingPolymerEngine::coupleRingPolymerBeads()
    {
        const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();
        const auto nAtoms = _ringPolymerBeads[0].getNumberOfAtoms();
        const auto temp =
            settings::ThermostatSettings::getActualTargetTemperature();

        // to silence narrowing conversion warning -- here not a problem
        const auto nBeads_ = static_cast<double>(nBeads);

        const auto rpmd_factor =
            RPMD_PREFACTOR * nBeads_ * nBeads_ * temp * temp;

        for (size_t i = 0; i < nBeads; ++i)
        {
            auto &bead1 = _ringPolymerBeads[i];
            auto &bead2 = _ringPolymerBeads[(i + 1) % nBeads];

            for (size_t j = 0; j < nAtoms; ++j)
            {
                auto &atom1 = bead1.getAtom(j);
                auto &atom2 = bead2.getAtom(j);

                const auto deltaPos = atom2.getPosition() - atom1.getPosition();

                const auto forceConstant = rpmd_factor * atom1.getMass();
                const auto force         = forceConstant * deltaPos;

                const auto energy = 0.5 * forceConstant * normSquared(deltaPos);

                _ringPolymerBeadsPhysicalData[i].addRingPolymerEnergy(energy);

                atom1.addForce(force);
                atom2.addForce(-force);
            }
        }
    }

    /**
     * @brief combining all beads into one simulation box
     *
     * @details coords, velocities and forces are averaged over all beads
     *
     */
    void RingPolymerEngine::combineBeads()
    {
        const auto numberOfBeads =
            settings::RingPolymerSettings::getNumberOfBeads();

        std::ranges::for_each(
            getSimulationBox().getAtoms(),
            [](auto &atom)
            {
                atom->setPosition({0.0, 0.0, 0.0});
                atom->setVelocity({0.0, 0.0, 0.0});
                atom->setForce({0.0, 0.0, 0.0});
            }
        );

        auto addCoordinates = [this, numberOfBeads](auto &bead)
        {
            for (size_t i = 0; i < bead.getNumberOfAtoms(); ++i)
            {
                auto      &atom   = bead.getAtom(i);
                const auto nBeads = static_cast<double>(numberOfBeads);

                const auto pos   = atom.getPosition() / nBeads;
                const auto vel   = atom.getVelocity() / nBeads;
                const auto force = atom.getForce() / nBeads;

                getSimulationBox().getAtom(i).addPosition(pos);
                getSimulationBox().getAtom(i).addVelocity(vel);
                getSimulationBox().getAtom(i).addForce(force);
            }
        };

        std::ranges::for_each(_ringPolymerBeads, addCoordinates);
    }

}   // namespace engine
