#include "engine.hpp"
#include "constants.hpp"

#include <vector>

using namespace std;

/**
 * @brief calculates the boxmomentum and sets it in the outputData
 *
 * @param simulationBox
 * @param outputData
 */
void Engine::calculateMomentum(SimulationBox &simulationBox, OutputData &outputData)
{
    vector<double> momentum(3, 0.0);

    for (auto &molecule : simulationBox._molecules)
    {

        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            auto velocities = molecule.getAtomVelocity(i);
            auto mass = molecule.getMass(i);

            momentum[0] += velocities[0] * mass / _S_TO_FS_;
            momentum[1] += velocities[1] * mass / _S_TO_FS_;
            momentum[2] += velocities[2] * mass / _S_TO_FS_;
        }
    }

    outputData.setMomentumVector(momentum);
}