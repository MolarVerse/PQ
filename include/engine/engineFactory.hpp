/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _ENGINE_FACTORY_HPP_
#define _ENGINE_FACTORY_HPP_

#include "engine.hpp"
#include "enums/jobtype.hpp"
#include "hessianEngine.hpp"
#include "mmmdEngine.hpp"
#include "optEngine.hpp"
#include "qmmdEngine.hpp"
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
                {JobType::QM_MD, [] { return std::make_unique<QMMDEngine>(); }},
                {JobType::RING_POLYMER_QM_MD,
                 [] { return std::make_unique<RingPolymerQMMDEngine>(); }},
                {JobType::QMMM_MD,
                 [] { return std::make_unique<QMMMMDEngine>(); }},
            };
}   // namespace engine

#endif   // _ENGINE_FACTORY_HPP_
