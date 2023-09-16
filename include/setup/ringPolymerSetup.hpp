#ifndef _RING_POLYMER_SETUP_HPP_

#define _RING_POLYMER_SETUP_HPP_

#include "ringPolymerEngine.hpp"

namespace setup
{
    void setupRingPolymer(engine::RingPolymerEngine &);

    /**
     * @class RingPolymerSetup
     *
     * @details class to setup a ring polymer simulation
     *
     */
    class RingPolymerSetup
    {
      private:
        engine::RingPolymerEngine &_engine;

      public:
        explicit RingPolymerSetup(engine::RingPolymerEngine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _RING_POLYMER_SETUP_HPP_