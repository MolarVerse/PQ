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
using namespace output;
using namespace simulationBox;
using namespace physicalData;
using std::make_unique;

/**
 * @brief constructor
 */
EngineOutput::EngineOutput()
{
    _energyOutput        = make_unique<EnergyOutput>("default.en");
    _instantEnergyOutput = make_unique<EnergyOutput>("default.instant_en");
    _momentumOutput      = make_unique<MomentumOutput>("default.mom");
    _xyzOutput           = make_unique<TrajectoryOutput>("default.xyz");
    _velOutput           = make_unique<TrajectoryOutput>("default.vel");
    _forceOutput         = make_unique<TrajectoryOutput>("default.force");
    _chargeOutput        = make_unique<TrajectoryOutput>("default.chg");
    _logOutput           = make_unique<LogOutput>("default.log");
    _stdoutOutput        = make_unique<StdoutOutput>("stdout");
    _rstFileOutput       = make_unique<RstFileOutput>("default.rst");
    _infoOutput          = make_unique<InfoOutput>("default.info");
    _virialOutput        = make_unique<VirialOutput>("default.vir");
    _stressOutput        = make_unique<StressOutput>("default.stress");
    _boxFileOutput       = make_unique<BoxFileOutput>("default.box");
    _optOutput           = make_unique<OptOutput>("default.opt");

    _rpmdRstFileOutput = make_unique<pq::RPMDRstFileOutput>("default.rpmd.rst");
    _rpmdXyzOutput     = make_unique<pq::RPMDTrajOutput>("default.rpmd.xyz");
    _rpmdVelOutput     = make_unique<pq::RPMDTrajOutput>("default.rpmd.vel");
    _rpmdForceOutput   = make_unique<pq::RPMDTrajOutput>("default.rpmd.force");
    _rpmdChargeOutput  = make_unique<pq::RPMDTrajOutput>("default.rpmd.chg");
    _rpmdEnergyOutput  = make_unique<pq::RPMDEnergyOutput>("default.rpmd.en");

    _timingsOutput = make_unique<TimingsOutput>("default.timings");
}

/**
 * @brief wrapper for energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeEnergyFile(
    const size_t        step,
    const PhysicalData &physicalData
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
    const size_t        step,
    const PhysicalData &physicalData
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
    const size_t        step,
    const PhysicalData &physicalData
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
void EngineOutput::writeXyzFile(SimulationBox &simulationBox)
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
void EngineOutput::writeVelFile(SimulationBox &simulationBox)
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
void EngineOutput::writeForceFile(SimulationBox &simulationBox)
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
void EngineOutput::writeChargeFile(SimulationBox &simulationBox)
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
    const double        time,
    const PhysicalData &physicalData
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
void EngineOutput::writeRstFile(SimulationBox &simulationBox, const size_t step)
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
    const size_t        step,
    const PhysicalData &physicalData
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
    const size_t        step,
    const PhysicalData &physicalData
)
{
    startTimingsSection("StressOutput");
    _stressOutput->write(step, physicalData);
    stopTimingsSection("StressOutput");
}

/**
 * @brief wrapper for box file output function
 *
 * @param step
 * @param simulationBox
 */
void EngineOutput::writeBoxFile(const size_t step, const Box &simulationBox)
{
    startTimingsSection("BoxFileOutput");
    _boxFileOutput->write(step, simulationBox);
    stopTimingsSection("BoxFileOutput");
}

/**
 * @brief wrapper for optimizer output function
 *
 * @param step
 * @param optimizer
 */
void EngineOutput::writeOptFile(
    const size_t         step,
    const pq::Optimizer &optimizer
)
{
    startTimingsSection("OptOutput");
    _optOutput->write(step, optimizer);
    stopTimingsSection("OptOutput");
}

/**
 * @brief wrapper for ring polymer restart file output function
 *
 * @param simulationBox
 * @param step
 */
void EngineOutput::writeRingPolymerRstFile(
    std::vector<SimulationBox> &beads,
    const size_t                step
)
{
    startTimingsSection("RingPolymerRestartFileOutput");
    _rpmdRstFileOutput->write(beads, step);
    stopTimingsSection("RingPolymerRestartFileOutput");
}

/**
 * @brief wrapper for ring polymer xyz file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerXyzFile(std::vector<SimulationBox> &beads)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _rpmdXyzOutput->writeXyz(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer velocity file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerVelFile(std::vector<SimulationBox> &beads)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _rpmdVelOutput->writeVelocities(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer force file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerForceFile(std::vector<SimulationBox> &beads)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _rpmdForceOutput->writeForces(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer charge file output function
 *
 * @param beads
 */
void EngineOutput::writeRingPolymerChargeFile(std::vector<SimulationBox> &beads)
{
    startTimingsSection("RingPolymerTrajectoryOutput");
    _rpmdChargeOutput->writeCharges(beads);
    stopTimingsSection("RingPolymerTrajectoryOutput");
}

/**
 * @brief wrapper for ring polymer energy file output function
 *
 * @param step
 * @param physicalData
 */
void EngineOutput::writeRingPolymerEnergyFile(
    const size_t                     step,
    const std::vector<PhysicalData> &dataVector
)
{
    startTimingsSection("RingPolymerEnergyOutput");
    _rpmdEnergyOutput->write(step, dataVector);
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

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief getter for energy output
 *
 * @return EnergyOutput
 */
EnergyOutput &EngineOutput::getEnergyOutput() { return *_energyOutput; }

/**
 * @brief getter for instant energy output
 *
 * @return EnergyOutput
 */
EnergyOutput &EngineOutput::getInstantEnergyOutput()
{
    return *_instantEnergyOutput;
}

/**
 * @brief getter for momentum output
 *
 * @return MomentumOutput
 */
MomentumOutput &EngineOutput::getMomentumOutput() { return *_momentumOutput; }

/**
 * @brief getter for xyz output
 *
 * @return TrajectoryOutput
 */
TrajectoryOutput &EngineOutput::getXyzOutput() { return *_xyzOutput; }

/**
 * @brief getter for velocity output
 *
 * @return TrajectoryOutput
 */
TrajectoryOutput &EngineOutput::getVelOutput() { return *_velOutput; }

/**
 * @brief getter for force output
 *
 * @return TrajectoryOutput
 */
TrajectoryOutput &EngineOutput::getForceOutput() { return *_forceOutput; }

/**
 * @brief getter for charge output
 *
 * @return TrajectoryOutput
 */
TrajectoryOutput &EngineOutput::getChargeOutput() { return *_chargeOutput; }

/**
 * @brief getter for log output
 *
 * @return LogOutput
 */
LogOutput &EngineOutput::getLogOutput() { return *_logOutput; }

/**
 * @brief getter for stdout output
 *
 * @return StdoutOutput
 */
StdoutOutput &EngineOutput::getStdoutOutput() { return *_stdoutOutput; }

/**
 * @brief getter for restart file output
 *
 * @return RstFileOutput
 */
RstFileOutput &EngineOutput::getRstFileOutput() { return *_rstFileOutput; }

/**
 * @brief getter for info output
 *
 * @return InfoOutput
 */
InfoOutput &EngineOutput::getInfoOutput() { return *_infoOutput; }

/**
 * @brief getter for virial output
 *
 * @return VirialOutput
 */
VirialOutput &EngineOutput::getVirialOutput() { return *_virialOutput; }

/**
 * @brief getter for stress output
 *
 * @return StressOutput
 */
StressOutput &EngineOutput::getStressOutput() { return *_stressOutput; }

/**
 * @brief getter for box file output
 *
 * @return BoxFileOutput
 */
BoxFileOutput &EngineOutput::getBoxFileOutput() { return *_boxFileOutput; }

/**
 * @brief getter for optimizer output
 *
 * @return OptOutput
 */
OptOutput &EngineOutput::getOptOutput() { return *_optOutput; }

/**
 * @brief getter for ring polymer restart file output
 *
 * @return RPMDRestartFileOutput
 */
RingPolymerRestartFileOutput &EngineOutput::getRingPolymerRstFileOutput()
{
    return *_rpmdRstFileOutput;
}

/**
 * @brief getter for ring polymer trajectory xyz output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerXyzOutput()
{
    return *_rpmdXyzOutput;
}

/**
 * @brief getter for ring polymer trajectory velocity output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerVelOutput()
{
    return *_rpmdVelOutput;
}

/**
 * @brief getter for ring polymer trajectory force output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerForceOutput()
{
    return *_rpmdForceOutput;
}

/**
 * @brief getter for ring polymer trajectory charge output
 *
 * @return RPMDTrajectoryOutput
 */
RingPolymerTrajectoryOutput &EngineOutput::getRingPolymerChargeOutput()
{
    return *_rpmdChargeOutput;
}

/**
 * @brief getter for ring polymer energy output
 *
 * @return RPMDEnergyOutput
 */
RingPolymerEnergyOutput &EngineOutput::getRingPolymerEnergyOutput()
{
    return *_rpmdEnergyOutput;
}

/**
 * @brief getter for timings output
 *
 * @return TimingsOutput
 */
TimingsOutput &EngineOutput::getTimingsOutput() { return *_timingsOutput; }