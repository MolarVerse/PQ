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

#ifndef _OPTIMIZER_HPP_

#define _OPTIMIZER_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr

namespace engine
{
    class OptEngine;   // forward declaration

}   // namespace engine

namespace optimization
{
    /**
     * @class Optimizer
     *
     * @brief Base class for all optimizers
     *
     */
    class Optimizer
    {
       protected:
        size_t _nEpochs;

       public:
        explicit Optimizer(const size_t);

        Optimizer()          = default;
        virtual ~Optimizer() = default;

        virtual void update(const double learningRate) = 0;
        virtual void updateLearningRate()              = 0;
        virtual bool checkConvergence()                = 0;
    };

}   // namespace optimization

#endif   // _OPTIMIZER_HPP_
