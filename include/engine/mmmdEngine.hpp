#ifndef _MM_MD_ENGINE_HPP_

#define _MM_MD_ENGINE_HPP_

#include "engine.hpp"

namespace engine
{

    /**
     * @class MmmdEngine
     *
     * @brief Contains all the information needed to run the simulation
     *
     */
    class MMMDEngine : public Engine
    {
      public:
        MMMDEngine()  = default;
        ~MMMDEngine() = default;

        void takeStep() override;
    };

}   // namespace engine

#endif   // _MM_MD_ENGINE_HPP_