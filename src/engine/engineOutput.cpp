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
 * @brief constructor
 */
EngineOutput::EngineOutput()
{
    _energyOutput        = std::make_unique<EnergyOutput>("default.en");
    _instantEnergyOutput = std::make_unique<EnergyOutput>("default.instant_en");
    _momentumOutput      = std::make_unique<MomentumOutput>("default.mom");
    _xyzOutput           = std::make_unique<TrajectoryOutput>("default.xyz");
    _velOutput           = std::make_unique<TrajectoryOutput>("default.vel");
    _forceOutput         = std::make_unique<TrajectoryOutput>("default.force");
    _chargeOutput        = std::make_unique<TrajectoryOutput>("default.chg");
    _logOutput           = std::make_unique<LogOutput>("default.log");
    _stdoutOutput        = std::make_unique<StdoutOutput>("stdout");
    _rstFileOutput       = std::make_unique<RstFileOutput>("default.rst");
    _infoOutput          = std::make_unique<InfoOutput>("default.info");
    _virialOutput        = std::make_unique<VirialOutput>("default.vir");
    _stressOutput        = std::make_unique<StressOutput>("default.stress");
    _boxFileOutput       = std::make_unique<BoxFileOutput>("default.box");

    _ringPolymerRstFileOutput =
        std::make_unique<RingPolymerRestartFileOutput>("default.rst");
    _ringPolymerXyzOutput =
        std::make_unique<RingPolymerTrajectoryOutput>("default.xyz");
    _ringPolymerVelOutput =
        std::make_unique<RingPolymerTrajectoryOutput>("default.vel");
    _ringPolymerForceOutput =
        std::make_unique<RingPolymerTrajectoryOutput>("default.force");
    _ringPolymerChargeOutput =
        std::make_unique<RingPolymerTrajectoryOutput>("default.chg");
    _ringPolymerEnergyOutput =
        std::make_unique<RingPolymerEnergyOutput>("default.en");

    _timingsOutput = std::make_unique<TimingsOutput>("default.timings");
}

/**
 * @brief wrapper for energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeEnergyFile(
    const size_t                      step,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("EnergyOutput");
    _energyOutput->write(step, physicalData);
    stopTimingsSection("EnergyOutput");
}

/**
 * @brief wrapper for instant energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeInstantEnergyFile(
    const size_t                      step,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("InstantEnergyOutput");
    _instantEnergyOutput->write(step, physicalData);
    stopTimingsSection("InstantEnergyOutput");
}

/**
 * @brief wrapper for momentum file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeMomentumFile(
    const size_t                      step,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("MomentumOutput");
    _momentumOutput->write(step, physicalData);
    stopTimingsSection("MomentumOutput");
}

/**
 * @brief wrapper for xyz file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeXyzFile(simulationBox::SimulationBox &simulationBox)
{
    startTimingsSection("TrajectoryOutput");
    _xyzOutput->writeXyz(simulationBox);
    stopTimingsSection("TrajectoryOutput");
}

/**
 * @brief wrapper for velocity file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeVelFile(simulationBox::SimulationBox &simulationBox)
{
    startTimingsSection("TrajectoryOutput");
    _velOutput->writeVelocities(simulationBox);
    stopTimingsSection("TrajectoryOutput");
}

/**
 * @brief wrapper for force file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeForceFile(simulationBox::SimulationBox &simulationBox)
{
    startTimingsSection("TrajectoryOutput");
    _forceOutput->writeForces(simulationBox);
    stopTimingsSection("TrajectoryOutput");
}

/**
 * @brief wrapper for charge file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeChargeFile(simulationBox::SimulationBox &simulationBox)
{
    startTimingsSection("TrajectoryOutput");
    _chargeOutput->writeCharges(simulationBox);
    stopTimingsSection("TrajectoryOutput");
}

/**
 * @brief wrapper for info file output function
 *
 * @param time
 * @param physicalData
 */
void EngineOutput::writeInfoFile(
    const double                      time,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("InfoOutput");
    _infoOutput->write(time, physicalData);
    stopTimingsSection("InfoOutput");
}

/**
 * @brief wrapper for restart file output function
 *
 * @param simulationBox
 * @param step
 */
void EngineOutput::writeRstFile(
    simulationBox::SimulationBox &simulationBox,
    const size_t                  step
)
{
    startTimingsSection("RstFileOutput");
    _rstFileOutput->write(simulationBox, step);
    stopTimingsSection("RstFileOutput");
}

/**
 * @brief wrapper for virial file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeVirialFile(
    const size_t                      step,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("VirialOutput");
    _virialOutput->write(step, physicalData);
    stopTimingsSection("VirialOutput");
}

/**
 * @brief wrapper for stress file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeStressFile(
    const size_t                      step,
    const physicalData::PhysicalData &physicalData
)
{
    startTimingsSection("StressOutput");
    _stressOutput->write(step, physicalData);
    stopTimingsSection("StressOutput");
}

/**
 * @brief wrapper for box file output function
 *
 * @param simulationBox
 */
void EngineOutput::writeBoxFile(
    const size_t              step,
    const simulationBox::Box &simulationBox
)
{
    startTimingsSection("BoxFileOutput");
    _boxFileOutput->write(step, simulationBox);
    stopTimingsSection("BoxFileOutput");
}

/**
 * @brief wrapper for ring polymer restart file output function
 *
 * @param simulationBox
 * @param step
 */
void EngineOutput::writeRingPolymerRstFile(
    std::vector<simulationBox::SimulationBox> &beads,
    const size_t                               step
)
{
    startTimingsSection("RingPolymerRestartFileOutput");
    _ringPolymerRstFileOutput->write(beads, step);
    stopTimingsSection("RingPolymerRestartFileOutput");
}

/**
 * @brief wrapper for ring polymer xyz file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerXyzFile(
    std::vector<simulationBox::SimulationBox> &beads
)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _ringPolymerXyzOutput->writeXyz(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer velocity file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerVelFile(
    std::vector<simulationBox::SimulationBox> &beads
)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _ringPolymerVelOutput->writeVelocities(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer force file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerForceFile(
    std::vector<simulationBox::SimulationBox> &beads
)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _ringPolymerForceOutput->writeForces(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer charge file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerChargeFile(
    std::vector<simulationBox::SimulationBox> &beads
)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _ringPolymerChargeOutput->writeCharges(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeRingPolymerEnergyFile(
    const size_t                                   step,
    const std::vector<physicalData::PhysicalData> &dataVector
)
{
    startTimingsSection("RingPolymerEnergyOutput");
    _ringPolymerEnergyOutput->write(step, dataVector);
    stopTimingsSection("RingPolymerEnergyOutput");
}

/**
 * @brief wrapper for timings file output function
 *
 * @param timer
 */
void EngineOutput::writeTimingsFile(timings::GlobalTimer &timer)
{
    // NOTE:
    // here is no timer applied, since the timings file is written at the end of
    // the simulation
    _timingsOutput->write(timer);
}

/**
 * @brief getter for ring polymer restart file output
 *
 * @return RPMDRestartFileOutput
 */
RingPolymerRestartFileOutput &EngineOutput::getRingPolymerRstFileOutput()
{
    return *_ringPolymerRstFileOutput;
}

/**
 * @brief getter for ring polymer trajectory xyz output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerXyzOutput()
{
    return *_ringPolymerXyzOutput;
}

/**
 * @brief getter for ring polymer trajectory velocity output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerVelOutput()
{
    return *_ringPolymerVelOutput;
}

/**
 * @brief getter for ring polymer trajectory force output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerForceOutput()
{
    return *_ringPolymerForceOutput;
}

/**
 * @brief getter for ring polymer trajectory charge output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerChargeOutput()
{
    return *_ringPolymerChargeOutput;
}

/**
 * @brief getter for ring polymer energy output
 *
 * @return RPMDEnergyOutput
 */
RingPolymerEnergyOutput &EngineOutput::getRingPolymerEnergyOutput()
{
    return *_ringPolymerEnergyOutput;
}