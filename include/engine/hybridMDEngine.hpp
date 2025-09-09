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

#ifndef _HYBRID_MD_ENGINE_HPP_

#define _HYBRID_MD_ENGINE_HPP_

#include "hybridConfigurator.hpp"
#include "mmmdEngine.hpp"
#include "qmCapable.hpp"

namespace engine
{
    /**
     * @brief HybridMDEngine
     *
     * @details This class is a pure virtual class that inherits from MMMDEngine
     * and QMCapable and is used to implement the Hybrid MD engine backbone
     * that can run in general combinations of MM and QM engines.
     *
     */
    class HybridMDEngine : virtual public MMMDEngine, public QMCapable
    {
       protected:
        configurator::HybridConfigurator _configurator{};

       public:
        HybridMDEngine()  = default;
        ~HybridMDEngine() = default;

        void calculateForces() override = 0;
    };

}   // namespace engine

#endif   // _HYBRID_MD_ENGINE_HPP_