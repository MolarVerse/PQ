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
 * @brief Output file for xyz, vel, force files
 *
 */
class output::TrajectoryOutput : public output::Output
{
  public:
    using output::Output::Output;

    void writexyz(simulationBox::SimulationBox &);
    void writeVelocities(simulationBox::SimulationBox &);
    void writeForces(simulationBox::SimulationBox &);
    void writeCharges(simulationBox::SimulationBox &);
};

#endif   // _TRAJECTORY_OUTPUT_HPP_