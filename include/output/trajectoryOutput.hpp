#ifndef _TRAJECTORY_OUTPUT_HPP_

#define _TRAJECTORY_OUTPUT_HPP_

#include "output.hpp"
#include "simulationBox.hpp"

namespace output
{
    class TrajectoryOutput;
}

/**
 * @class TrajectoryOutput inherits from Output
 *
 * @brief Output for xyz, vel, force, charges files
 *
 */
class output::TrajectoryOutput : public output::Output
{
  public:
    using output::Output::Output;

    void writeXyz(simulationBox::SimulationBox &);
    void writeVelocities(simulationBox::SimulationBox &);
    void writeForces(simulationBox::SimulationBox &);
    void writeCharges(simulationBox::SimulationBox &);
};

#endif   // _TRAJECTORY_OUTPUT_HPP_