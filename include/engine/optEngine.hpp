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

#include "engine.hpp"                 // for Engine
#include "evaluator.hpp"              // for Evaluator
#include "learningRateStrategy.hpp"   // for learningRateStrategy
#include "optimizer.hpp"              // for Optimizer
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
        std::shared_ptr<pq::Optimizer>    _optimizer;
        std::shared_ptr<pq::LearningRate> _learningRateStrategy;
        std::shared_ptr<pq::Evaluator>    _evaluator;

        pq::SharedPhysicalData _physicalDataOld;

        bool _converged  = false;
        bool _optStopped = false;

       public:
        void run() final;
        void takeStep() final;
        void writeOutput() final;

        /***************************
         * standard setter methods *
         ***************************/

        void setOptimizer(const pq::Optimizer &optimizer);
        void setLearningRateStrategy(const pq::LearningRate &);
        void setEvaluator(const pq::Evaluator &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::shared_ptr<pq::Optimizer> &getOptimizer();
        [[nodiscard]] std::shared_ptr<pq::LearningRate> &getLearningRateStrategy(
        );
        [[nodiscard]] std::shared_ptr<pq::Evaluator> &getEvaluator();

        [[nodiscard]] pq::PhysicalData &getPhysicalDataOld();

        /***************************************
         * standard make smart pointer methods *
         ***************************************/

        template <typename T>
        void makeOptimizer(T optimizer);

        template <typename T>
        void makeLearningRateStrategy(T strategy);

        template <typename T>
        void makeEvaluator(T evaluator);
    };

}   // namespace engine

#include "optEngine.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _OPT_ENGINE_HPP_