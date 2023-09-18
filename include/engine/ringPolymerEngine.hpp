#ifndef _RING_POLYMER_ENGINE_HPP_

#define _RING_POLYMER_ENGINE_HPP_

#include "engine.hpp"

#include <vector>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace engine
{
    /**
     * @class RingPolymerEngine
     *
     * @details Contains all the information needed to run a ring polymer simulation
     *
     */
    class RingPolymerEngine : virtual public Engine
    {
      protected:
        std::vector<simulationBox::SimulationBox> _ringPolymerBeads;

      public:
        void writeOutput() override;

        void addRingPolymerBead(const simulationBox::SimulationBox &bead) { _ringPolymerBeads.push_back(bead); }
        void coupleRingPolymerBeads();
        void combineBeads();
    };
}   // namespace engine

#endif   // _RING_POLYMER_ENGINE_HPP_