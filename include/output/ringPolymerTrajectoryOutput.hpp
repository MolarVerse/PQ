#ifndef _RING_POLYMER_TRAJECTORY_OUTPUT_HPP_

#define _RING_POLYMER_TRAJECTORY_OUTPUT_HPP_

#include "output.hpp"

#include <vector>   // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace output
{
    /**
     * @class RingPolymerTrajectoryOutput inherits from Output
     *
     * @brief Output for xyz, vel, force, charges files for all ring polymer beads
     *
     */
    class RingPolymerTrajectoryOutput : public Output
    {
      public:
        using Output::Output;

        void writeHeader(const simulationBox::SimulationBox &);
        void writeXyz(std::vector<simulationBox::SimulationBox> &);
        void writeVelocities(std::vector<simulationBox::SimulationBox> &){};
        void writeForces(std::vector<simulationBox::SimulationBox> &){};
        void writeCharges(std::vector<simulationBox::SimulationBox> &){};
    };
}   // namespace output

#endif   // _RING_POLYMER_TRAJECTORY_OUTPUT_HPP_