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

#ifndef _HESSIAN_ENGINE_HPP_

#define _HESSIAN_ENGINE_HPP_

#include <memory>

#include "engine.hpp"
#include "typeAliases.hpp"

namespace engine
{
    class HessianEngine : public Engine
    {
       private:
        pq::SharedPhysicalData _physicalDataOld =
            std::make_shared<pq::PhysicalData>();

        pq::SharedOptimizer    _optimizer;
        pq::SharedLearningRate _learningRateStrategy;
        pq::SharedEvaluator    _evaluator;

        bool _converged  = false;
        bool _optStopped = false;

        [[nodiscard]] pq::SharedEvaluator setupEvaluator();
        [[nodiscard]] pq::SharedHessianBuilder setupHessianBuilder() const;

        void setupOptimization(const pq::SharedEvaluator &evaluator);
        void runOptimization();
        void takeOptimizationStep();
        void writeOptimizationOutput();

        [[nodiscard]] pq::SharedOptimizer    setupEmptyOptimizer();
        [[nodiscard]] pq::SharedLearningRate setupLearningRateStrategy();
        void setupConvergence(pq::SharedOptimizer &optimizer);
        void setupMinMaxLearningRate(pq::SharedLearningRate &learningRate);
        void writeOptimizationSetupInfo();

        void writeHessian(const pq::HessianMatrix &hessian) const;
        void writeHessianInfo(const pq::HessianMatrix &hessian) const;
        void addTimers();

       public:
        void run() final;
        void writeOutput() final;

        [[nodiscard]] pq::SharedPhysicalData getSharedPhysicalDataOld();
        [[nodiscard]] output::OptOutput     &getOptOutput();
    };

}   // namespace engine

#endif   // _HESSIAN_ENGINE_HPP_
