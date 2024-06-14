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

#ifndef _OPT_ENGINE_HPP_

#define _OPT_ENGINE_HPP_

#include <memory>   // for unique_ptr

#include "engine.hpp"      // for Engine
#include "optimizer.hpp"   // for Optimizer

namespace engine
{

    /**
     * @class OptEngine
     *
     * @brief Optimizer engine
     *
     */
    class OptEngine : public Engine
    {
       private:
        std::shared_ptr<optimization::Optimizer> _optimizer;

       public:
        void setOptimizer(std::shared_ptr<optimization::Optimizer> &optimizer);

        template <typename T>
        void makeOptimizer(T optimizer);
    };

}   // namespace engine

#include "optEngine.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _OPT_ENGINE_HPP_