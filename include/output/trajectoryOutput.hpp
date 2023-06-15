#ifndef _TRAJECTORY_OUTPUT_HPP_

#define _TRAJECTORY_OUTPUT_HPP_

#include "output.hpp"
#include "simulationBox.hpp"

/**
 * @class TrajectoryOutput inherits from Output
 *
 * @brief Output file for xyz, vel, force files
 *
 */
class TrajectoryOutput : public Output
{
public:
    using Output::Output;

    void writexyz(simulationBox::SimulationBox &);
    void writeVelocities(simulationBox::SimulationBox &);
    void writeForces(simulationBox::SimulationBox &);
    void writeCharges(simulationBox::SimulationBox &);
};

#endif // _TRAJECTORY_OUTPUT_HPP_