#include "qmRunner.hpp"

using QM::QMRunner;

/**
 * @brief run the qm engine
 *
 * @param box
 */
void QMRunner::run(simulationBox::SimulationBox &box)
{
    writeCoordsFile(box);
    execute();
}