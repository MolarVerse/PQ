#include "qmRunner.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

using QM::QMRunner;

/**
 * @brief run the qm engine
 *
 * @param box
 */
void QMRunner::run(simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    writeCoordsFile(box);
    execute();
    readForceFile(box, physicalData);
}