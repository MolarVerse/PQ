#ifndef _RING_POLYMER_QM_MD_ENGINE_HPP_

#define _RING_POLYMER_QM_MD_ENGINE_HPP_

#include "qmmdEngine.hpp"
#include "ringPolymerEngine.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace engine
{
    /**
     * @class RingPolymerQMMDEngine
     *
     * @details Contains all the information needed to run a ring polymer QM MD simulation
     *
     */
    class RingPolymerQMMDEngine : public QMMDEngine, public RingPolymerEngine
    {
      public:
        void takeStep() override{};
    };
}   // namespace engine

#endif   // _RING_POLYMER_QM_MD_ENGINE_HPP_