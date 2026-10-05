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
#include "settings.hpp"

namespace engine
{
    /**
     * @brief Engine factory
     *
     * @details This is a static unordered map that maps JobType to a function
     * that creates a unique pointer to the corresponding Engine with the given
     * Settings.
     *
     */
    static const std::unordered_map<
        JobType,
        std::function<std::unique_ptr<Engine>(Settings settings)>>
        engineFactory{
            {JobType::MM_OPT,
             [](Settings settings)
             { return std::make_unique<OptEngine>(settings); }},
            {JobType::MM_HESSIAN,
             [](Settings settings)
             { return std::make_unique<HessianEngine>(settings); }},
            {JobType::MM_MD,
             [](Settings settings)
             { return std::make_unique<MMMDEngine>(settings); }},
            {JobType::QM_MD,
             [](Settings settings)
             { return std::make_unique<QMMDEngine>(settings); }},
            {JobType::RING_POLYMER_QM_MD,
             [](Settings settings)
             { return std::make_unique<RingPolymerQMMDEngine>(settings); }},
            {JobType::QMMM_MD,
             [](Settings settings)
             { return std::make_unique<QMMMMDEngine>(settings); }},
        };
}   // namespace engine

#endif   // _ENGINE_FACTORY_HPP_
