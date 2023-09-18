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