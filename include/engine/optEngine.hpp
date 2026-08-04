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

#include "engine.hpp"
#include "evaluator.hpp"
#include "learningRateStrategy.hpp"
#include "optimizer.hpp"

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
        std::shared_ptr<opt::Optimizer>            _optimizer;
        std::shared_ptr<opt::LearningRateStrategy> _learningRateStrategy;
        std::shared_ptr<opt::Evaluator>            _evaluator;

        std::shared_ptr<physicalData::PhysicalData> _physicalDataOld =
            std::make_shared<physicalData::PhysicalData>();

        bool _converged  = false;
        bool _optStopped = false;

       public:
        void run() final;
        void takeStep();
        void writeOutput() final;

        /***************************
         * standard setter methods *
         ***************************/

        void setOptimizer(const std::shared_ptr<opt::Optimizer>);
        void setLearningRateStrategy(
            const std::shared_ptr<opt::LearningRateStrategy>
        );
        void setEvaluator(const std::shared_ptr<opt::Evaluator>);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] opt::Optimizer            &getOptimizer();
        [[nodiscard]] opt::LearningRateStrategy &getLearningRate();
        [[nodiscard]] opt::Evaluator            &getEvaluator();
        [[nodiscard]] opt::Convergence          &getConvergence();

        // clang-format off
        [[nodiscard]] std::shared_ptr<opt::Optimizer> getSharedOptimizer();
        [[nodiscard]] std::shared_ptr<opt::LearningRateStrategy> getSharedLearningRate();
        [[nodiscard]] std::shared_ptr<opt::Evaluator> getSharedEvaluator();
        
        [[nodiscard]] physicalData::PhysicalData &getPhysicalDataOld();
        [[nodiscard]] std::shared_ptr<physicalData::PhysicalData> getSharedPhysicalDataOld();
        // clang-format on

        [[nodiscard]] output::OptOutput &getOptOutput();
    };

}   // namespace engine

#endif   // _OPT_ENGINE_HPP_
