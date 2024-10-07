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

#include <algorithm>    // for __for_each_fn
#include <cstddef>      // for size_t
#include <functional>   // for identity

#include "atom.hpp"                                  // for Atom
#include "constants/internalConversionFactors.hpp"   // for _RPMD_PREFACTOR_
#include "engineOutput.hpp"                          // for EngineOutput
#include "outputFileSettings.hpp"                    // for OutputFileSettings
#include "physicalData.hpp"                          // for PhysicalData
#include "ringPolymerSettings.hpp"                   // for RingPolymerSettings
#include "thermostatSettings.hpp"                    // for ThermostatSettings
#include "timer.hpp"                                 // for Timings
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"   // for Vector3D, normSquared

using engine::Engine;
using engine::RingPolymerEngine;

using namespace settings;
using namespace constants;
using namespace physicalData;

/**
 * @brief resizes the vector of physical data for the ring polymer beads
 *
 * @param physicalData
 */
void RingPolymerEngine::resizeRingPolymerBeadPhysicalData(
    const size_t numberOfBeads
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

    const auto outputFreq = OutputFileSettings::getOutputFrequency();
    const auto step0      = TimingsSettings::getStepCount();
    const auto effStep    = _step + step0;

    if (0 == _step % outputFreq)
    {
        _engineOutput.writeXyzFile(*_simulationBox);
        _engineOutput.writeVelFile(*_simulationBox);
        _engineOutput.writeForceFile(*_simulationBox);
        _engineOutput.writeChargeFile(*_simulationBox);
        _engineOutput.writeRstFile(*_simulationBox, *_thermostat, effStep);

        _engineOutput.writeRingPolymerRstFile(_ringPolymerBeads, effStep);
        _engineOutput.writeRingPolymerXyzFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerVelFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerForceFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerChargeFile(_ringPolymerBeads);
    }

    // NOTE:
    // stop and restart immediately time manager - maximum lost time is en
    // file writing in last step of simulation but on the other hand setup
    // is now included in total simulation time Unfortunately, setup is
    // therefore included in the first looptime output but this is not a big
    // problem - could also be a feature and not a bug
    _timer.stopSimulationTimer();
    _timer.startSimulationTimer();

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        rpmdData[i].setLoopTime(_timer.calculateLoopTime());
        averageRPMDData[i].updateAverages(rpmdData[i]);
    }

    if (0 == _step % outputFreq)
    {
        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
            averageRPMDData[i].makeAverages(static_cast<double>(outputFreq));

        _physicalData->copy(mean(rpmdData));
        _averagePhysicalData = mean(averageRPMDData);

        const auto dt            = TimingsSettings::getTimeStep();
        const auto effStepDouble = static_cast<double>(effStep);
        const auto simTime       = effStepDouble * dt * _FS_TO_PS_;

        _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
        _engineOutput.writeInstantEnergyFile(effStep, *_physicalData);
        _engineOutput.writeMomentumFile(effStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(simTime, _averagePhysicalData);

        _engineOutput.writeRingPolymerEnergyFile(
            step0 + _step,
            averageRPMDData
        );

        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
            averageRPMDData[i] = PhysicalData();

        _averagePhysicalData = PhysicalData();
    }

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i) rpmdData[i].reset();
}

/**
 * @brief coupling step of ring polymers
 *
 */
void RingPolymerEngine::coupleRingPolymerBeads()
{
    const auto nBeads      = RingPolymerSettings::getNumberOfBeads();
    const auto nAtoms      = _ringPolymerBeads[0].getNumberOfAtoms();
    const auto temp        = ThermostatSettings::getActualTargetTemperature();
    const auto rpmd_factor = _RPMD_PREFACTOR_ * nBeads * nBeads * temp * temp;

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
    const auto numberOfBeads = RingPolymerSettings::getNumberOfBeads();

    std::ranges::for_each(
        _simulationBox->getAtoms(),
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
            auto &atom = bead.getAtom(i);

            const auto pos   = atom.getPosition() / double(numberOfBeads);
            const auto vel   = atom.getVelocity() / double(numberOfBeads);
            const auto force = atom.getForce() / double(numberOfBeads);

            _simulationBox->getAtom(i).addPosition(pos);
            _simulationBox->getAtom(i).addVelocity(vel);
            _simulationBox->getAtom(i).addForce(force);
        }
    };

    std::ranges::for_each(_ringPolymerBeads, addCoordinates);
}
