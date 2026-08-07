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

#include "constant.hpp"               // for ConstantLearningRateStrategy
#include "engine.hpp"                 // for Engine
#include "evaluator.hpp"              // for Evaluator
#include "learningRateStrategy.hpp"   // for learningRateStrategy
#include "mmEvaluator.hpp"            // for MMEvaluator
#include "optimizer.hpp"              // for Optimizer
#include "steepestDescent.hpp"        // for SteepestDescent
#include "typeAliases.hpp"

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
        pq::SharedOptimizer    _optimizer;
        pq::SharedLearningRate _learningRateStrategy;
        pq::SharedEvaluator    _evaluator;

        pq::SharedPhysicalData _physicalDataOld =
            std::make_shared<pq::PhysicalData>();

        bool _converged  = false;
        bool _optStopped = false;

       public:
        void run() final;
        void takeStep();
        void writeOutput() final;

        /***************************
         * standard setter methods *
         ***************************/

        void setOptimizer(const std::shared_ptr<pq::Optimizer>);
        void setLearningRateStrategy(const std::shared_ptr<pq::LearningRate>);
        void setEvaluator(const std::shared_ptr<pq::Evaluator>);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::Optimizer    &getOptimizer();
        [[nodiscard]] pq::LearningRate &getLearningRate();
        [[nodiscard]] pq::Evaluator    &getEvaluator();
        [[nodiscard]] pq::Convergence  &getConvergence();

        [[nodiscard]] std::shared_ptr<pq::Optimizer>    getSharedOptimizer();
        [[nodiscard]] std::shared_ptr<pq::LearningRate> getSharedLearningRate();
        [[nodiscard]] std::shared_ptr<pq::Evaluator>    getSharedEvaluator();

        [[nodiscard]] pq::PhysicalData      &getPhysicalDataOld();
        [[nodiscard]] pq::SharedPhysicalData getSharedPhysicalDataOld();

        [[nodiscard]] output::OptOutput &getOptOutput();
    };

}   // namespace engine

#endif   // _OPT_ENGINE_HPP_