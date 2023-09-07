#ifndef _TRAJECTORY_OUTPUT_HPP_

#define _TRAJECTORY_OUTPUT_HPP_

#include "output.hpp"   // for Output

namespace simulationBox
{
    class SimulationBox;
}   // namespace simulationBox

namespace output
{
    /**
     * @class TrajectoryOutput inherits from Output
     *
     * @brief Output for xyz, vel, force, charges files
     *
     */
    class TrajectoryOutput : public Output
    {
      public:
        using Output::Output;

        void writeHeader(const simulationBox::SimulationBox &);
        void writeXyz(simulationBox::SimulationBox &);
        void writeVelocities(simulationBox::SimulationBox &);
        void writeForces(simulationBox::SimulationBox &);
        void writeCharges(simulationBox::SimulationBox &);
    };

}   // namespace output

#endif   // _TRAJECTORY_OUTPUT_HPP_