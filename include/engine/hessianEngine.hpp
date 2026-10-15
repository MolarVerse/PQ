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
#include "hessianBuilder.hpp"
#include "learningRateStrategy.hpp"

namespace engine
{
    class HessianEngine : public Engine
    {
       private:
        std::shared_ptr<physicalData::PhysicalData> _physicalDataOld =
            std::make_shared<physicalData::PhysicalData>();

        std::shared_ptr<opt::Optimizer>            _optimizer;
        std::shared_ptr<opt::LearningRateStrategy> _learningRateStrategy;
        std::shared_ptr<opt::Evaluator>            _evaluator;

        bool _converged  = false;
        bool _optStopped = false;

       public:
        using Engine::Engine;

        void run() override;
        void writeOutput() override;

        [[nodiscard]]
        std::shared_ptr<physicalData::PhysicalData> getSharedPhysicalDataOld();
        [[nodiscard]] out::OptOutput               &getOptOutput();

       private:
        [[nodiscard]]
        std::shared_ptr<opt::Evaluator> _setupEvaluator();

        void _setupOptimization(
            const std::shared_ptr<opt::Evaluator> &evaluator
        );
        void _runOptimization();
        void _takeOptimizationStep();
        void _writeOptimizationOutput();

        [[nodiscard]]
        std::shared_ptr<opt::Optimizer> _setupEmptyOptimizer();

        void _writeOptimizationSetupInfo();
    };

}   // namespace engine

#endif   // _HESSIAN_ENGINE_HPP_
