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

#ifndef _OPTIMIZER_SETUP_HPP_

#define _OPTIMIZER_SETUP_HPP_

#include <memory>   // for shared_ptr

namespace engine
{
    class Engine;      // Forward declaration
    class OptEngine;   // Forward declaration

}   // namespace engine

namespace optimization
{
    class Optimizer;              // Forward declaration
    class LearningRateStrategy;   // Forward declaration

}   // namespace optimization

namespace setup
{
    using Optimizer            = optimization::Optimizer;
    using LearningRateStrategy = optimization::LearningRateStrategy;

    void setupOptimizer(engine::Engine &);

    /**
     * @class OptimizerSetup
     *
     * @brief Setup optimizer
     *
     */
    class OptimizerSetup
    {
       protected:
        engine::OptEngine &_optEngine;

       public:
        explicit OptimizerSetup(engine::OptEngine &optEngine);

        void setup();

        std::shared_ptr<Optimizer>            setupEmptyOptimizer();
        std::shared_ptr<LearningRateStrategy> setupLearningRateStrategy();
    };

}   // namespace setup

#endif   // _OPTIMIZER_SETUP_HPP_