#ifndef _MM_MD_ENGINE_HPP_

#define _MM_MD_ENGINE_HPP_

#include "engine.hpp"

namespace engine
{

    /**
     * @class MMMDEngine
     *
     * @brief Contains all the information needed to run an MM MD simulation
     *
     */
    class MMMDEngine : public Engine
    {
      public:
        void takeStep() override;
    };

}   // namespace engine

#endif   // _MM_MD_ENGINE_HPP_