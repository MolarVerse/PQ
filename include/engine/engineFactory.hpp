#ifndef _ENGINE_FACTORY_HPP_
#define _ENGINE_FACTORY_HPP_

#include "engine.hpp"
#include "enums/jobtype.hpp"
#include "hessianEngine.hpp"
#include "mmmdEngine.hpp"
#include "optEngine.hpp"
#include "qmmmMDEngine.hpp"
#include "ringPolymerqmmdEngine.hpp"

namespace engine
{
    static const std::
        unordered_map<JobType, std::function<std::unique_ptr<Engine>()>>
            engineFactory{
                {JobType::MM_OPT, [] { return std::make_unique<OptEngine>(); }},
                {JobType::MM_HESSIAN,
                 [] { return std::make_unique<HessianEngine>(); }},
                {JobType::MM_MD, [] { return std::make_unique<MMMDEngine>(); }},
                {JobType::QM_MD, [] { return std::make_unique<MMMDEngine>(); }},
                {JobType::RING_POLYMER_QM_MD,
                 [] { return std::make_unique<RingPolymerQMMDEngine>(); }},
                {JobType::QMMM_MD,
                 [] { return std::make_unique<QMMMMDEngine>(); }},
            };
}   // namespace engine

#endif   // _ENGINE_FACTORY_HPP_
