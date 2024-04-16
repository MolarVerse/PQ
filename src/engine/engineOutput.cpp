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

#include "engineOutput.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}
namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

using namespace engine;

/**
 * @brief wrapper for energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeEnergyFile(const size_t step, const double loopTime, const physicalData::PhysicalData &physicalData)
{
    _energyOutput->write(step, loopTime, physicalData);
}

/**
 * @brief wrapper for momentum file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeMomentumFile(const size_t step, const physicalData::PhysicalData &physicalData)
{
    _momentumOutput->write(step, physicalData);
}

/**
 * @brief wrapper for xyz file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeXyzFile(simulationBox::SimulationBox &simulationBox) { _xyzOutput->writeXyz(simulationBox); }

/**
 * @brief wrapper for velocity file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeVelFile(simulationBox::SimulationBox &simulationBox) { _velOutput->writeVelocities(simulationBox); }

/**
 * @brief wrapper for force file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeForceFile(simulationBox::SimulationBox &simulationBox) { _forceOutput->writeForces(simulationBox); }

/**
 * @brief wrapper for charge file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeChargeFile(simulationBox::SimulationBox &simulationBox) { _chargeOutput->writeCharges(simulationBox); }

/**
 * @brief wrapper for info file output function
 *
 * @param time
 * @param physicalData
 */
void EngineOutput::writeInfoFile(const double time, const double loopTime, const physicalData::PhysicalData &physicalData)
{
    _infoOutput->write(time, loopTime, physicalData);
}

/**
 * @brief wrapper for restart file output function
 *
 * @param simulationBox
 * @param step
 */
void EngineOutput::writeRstFile(simulationBox::SimulationBox &simulationBox, const size_t step)
{
    _rstFileOutput->write(simulationBox, step);
}

/**
 * @brief wrapper for virial file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeVirialFile(const size_t step, const physicalData::PhysicalData &physicalData)
{
    _virialOutput->write(step, physicalData);
}

/**
 * @brief wrapper for stress file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeStressFile(const size_t step, const physicalData::PhysicalData &physicalData)
{
    _stressOutput->write(step, physicalData);
}

/**
 * @brief wrapper for ring polymer restart file output function
 *
 * @param simulationBox
 * @param step
 */
void EngineOutput::writeRingPolymerRstFile(std::vector<simulationBox::SimulationBox> &beads, const size_t step)
{
    _ringPolymerRstFileOutput->write(beads, step);
}

/**
 * @brief wrapper for ring polymer xyz file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerXyzFile(std::vector<simulationBox::SimulationBox> &beads)
{
    _ringPolymerXyzOutput->writeXyz(beads);
}

/**
 * @brief wrapper for ring polymer velocity file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerVelFile(std::vector<simulationBox::SimulationBox> &beads)
{
    _ringPolymerVelOutput->writeVelocities(beads);
}

/**
 * @brief wrapper for ring polymer force file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerForceFile(std::vector<simulationBox::SimulationBox> &beads)
{
    _ringPolymerForceOutput->writeForces(beads);
}

/**
 * @brief wrapper for ring polymer charge file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerChargeFile(std::vector<simulationBox::SimulationBox> &beads)
{
    _ringPolymerChargeOutput->writeCharges(beads);
}

/**
 * @brief wrapper for ring polymer energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeRingPolymerEnergyFile(const size_t step, const std::vector<physicalData::PhysicalData> &dataVector)
{
    _ringPolymerEnergyOutput->write(step, dataVector);
}