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

#include "ringPolymerEngine.hpp"

#include "atom.hpp"                                  // for Atom
#include "constants/internalConversionFactors.hpp"   // for _RPMD_PREFACTOR_
#include "engineOutput.hpp"                          // for EngineOutput
#include "outputFileSettings.hpp"                    // for OutputFileSettings
#include "physicalData.hpp"                          // for PhysicalData
#include "ringPolymerSettings.hpp"                   // for RingPolymerSettings
#include "thermostatSettings.hpp"                    // for ThermostatSettings
#include "timings.hpp"                               // for Timings
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for Vector3D, normSquared

#include <algorithm>    // for __for_each_fn
#include <cstddef>      // for size_t
#include <functional>   // for identity

using engine::Engine;
using engine::RingPolymerEngine;

/**
 * @brief resizes the vector of physical data for the ring polymer beads
 *
 * @param physicalData
 */
void RingPolymerEngine::resizeRingPolymerBeadPhysicalData(const size_t numberOfBeads)
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

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        _averageRingPolymerBeadsPhysicalData[i].updateAverages(_ringPolymerBeadsPhysicalData[i]);
        _ringPolymerBeadsPhysicalData[i].reset();
    }

    const auto outputFrequency = settings::OutputFileSettings::getOutputFrequency();

    if (0 == _step % outputFrequency)
    {
        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
            _averageRingPolymerBeadsPhysicalData[i].makeAverages(static_cast<double>(outputFrequency));

        _averagePhysicalData = mean(_averageRingPolymerBeadsPhysicalData);

        const auto dt             = settings::TimingsSettings::getTimeStep();
        const auto step0          = _timings.getStepCount();
        const auto effectiveStep  = _step + step0;
        const auto simulationTime = static_cast<double>(effectiveStep) * dt * constants::_FS_TO_PS_;
        const auto loopTime       = _timings.calculateLoopTime(_step);

        _engineOutput.writeEnergyFile(effectiveStep, loopTime, _averagePhysicalData);
        _engineOutput.writeMomentumFile(effectiveStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(simulationTime, loopTime, _averagePhysicalData);
        _engineOutput.writeXyzFile(_simulationBox);
        _engineOutput.writeVelFile(_simulationBox);
        _engineOutput.writeForceFile(_simulationBox);
        _engineOutput.writeChargeFile(_simulationBox);
        _engineOutput.writeRstFile(_simulationBox, _step + step0);

        _engineOutput.writeRingPolymerRstFile(_ringPolymerBeads, step0 + _step);
        _engineOutput.writeRingPolymerXyzFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerVelFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerForceFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerChargeFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerEnergyFile(step0 + _step, _averageRingPolymerBeadsPhysicalData);

        for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
            _averageRingPolymerBeadsPhysicalData[i] = physicalData::PhysicalData();

        _averagePhysicalData = physicalData::PhysicalData();
    }
}

/**
 * @brief coupling step of ring polymers
 *
 */
void RingPolymerEngine::coupleRingPolymerBeads()
{
    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();
    const auto numberOfAtoms = _ringPolymerBeads[0].getNumberOfAtoms();
    const auto temperature   = settings::ThermostatSettings::getTargetTemperature();
    const auto rpmd_factor   = constants::_RPMD_PREFACTOR_ * numberOfBeads * numberOfBeads * temperature * temperature;

    for (size_t i = 0; i < numberOfBeads; ++i)
    {
        auto &bead1 = _ringPolymerBeads[i];
        auto &bead2 = _ringPolymerBeads[(i + 1) % numberOfBeads];

        for (size_t j = 0; j < numberOfAtoms; ++j)
        {
            auto &atom1 = bead1.getAtom(j);
            auto &atom2 = bead2.getAtom(j);

            const auto deltaPosition = atom2.getPosition() - atom1.getPosition();

            const auto forceConstant = rpmd_factor * atom1.getMass();
            const auto force         = forceConstant * deltaPosition;

            _ringPolymerBeadsPhysicalData[i].addRingPolymerEnergy(0.5 * forceConstant * normSquared(deltaPosition));

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
    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();

    std::ranges::for_each(_simulationBox.getAtoms(),
                          [](auto &atom)
                          {
                              atom->setPosition({0.0, 0.0, 0.0});
                              atom->setVelocity({0.0, 0.0, 0.0});
                              atom->setForce({0.0, 0.0, 0.0});
                          });

    auto addCoordinates = [this, numberOfBeads](auto &bead)
    {
        for (size_t i = 0; i < bead.getNumberOfAtoms(); ++i)
        {
            auto &atom = bead.getAtom(i);

            _simulationBox.getAtom(i).addPosition(atom.getPosition() / double(numberOfBeads));
            _simulationBox.getAtom(i).addVelocity(atom.getVelocity() / double(numberOfBeads));
            _simulationBox.getAtom(i).addForce(atom.getForce() / double(numberOfBeads));
        }
    };

    std::ranges::for_each(_ringPolymerBeads, addCoordinates);
}