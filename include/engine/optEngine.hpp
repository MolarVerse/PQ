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

#ifndef ENGINE__OPT_ENGINE_HPP
#define ENGINE__OPT_ENGINE_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "constraints.hpp"
#include "engine.hpp"
#include "evaluator.hpp"
#include "intraNonBonded.hpp"
#include "learningRateStrategy.hpp"
#include "optimizer.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"

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

        bool _converged  = false;
        bool _optStopped = false;

       public:
        using Engine::Engine;

        void run() override;
        void takeStep();
        void writeOutput(
            JobType                   jobType,
            const OutputFileSettings &outputSettings
        ) override;

        static void throwIfStopped(
            bool                            optStopped,
            size_t                          step,
            size_t                          nEpochs,
            const std::vector<std::string> &errorMessages
        );
        static void throwOnFailure(
            bool                            converged,
            bool                            optStopped,
            size_t                          step,
            size_t                          nEpochs,
            const std::vector<std::string> &errorMessages
        );
        [[nodiscard]] static std::string  notConvergedMessage(size_t nEpochs);
        [[nodiscard]] static EnergyPerMol optimizedEnergy(
            const physicalData::PhysicalData &physicalData
        );
        [[nodiscard]] static EnergyPerMol evaluateConstrained(
            opt::Evaluator                 &evaluator,
            constraints::Constraints       &constraints,
            molsys::SimulationBox          &simulationBox,
            intraNonBonded::IntraNonBonded &intraNonBonded,
            physicalData::PhysicalData     &physicalData
        );

        /***************************
         * standard setter methods *
         ***************************/

        void setOptimizer(const std::shared_ptr<opt::Optimizer> &optimizer);
        void setLearningRateStrategy(
            const std::shared_ptr<opt::LearningRateStrategy>
                &learningRateStrategy
        );
        void setEvaluator(const std::shared_ptr<opt::Evaluator> &evaluator);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] opt::Optimizer            &getOptimizer();
        [[nodiscard]] opt::LearningRateStrategy &getLearningRate();
        [[nodiscard]] opt::Evaluator            &getEvaluator();
        [[nodiscard]] opt::Convergence          &getConvergence();

        [[nodiscard]] std::shared_ptr<opt::Optimizer> getSharedOptimizer();
        [[nodiscard]] std::shared_ptr<opt::LearningRateStrategy> getSharedLearningRate(
        );
        [[nodiscard]] std::shared_ptr<opt::Evaluator> getSharedEvaluator();

        [[nodiscard]] out::OptOutput &getOptOutput();

       private:
        void _evaluate();
        void _writeOutput(
            const OutputFileSettings &outputSettings,
            size_t                    startingStepCount
        );
    };

}   // namespace engine

#endif   // ENGINE__OPT_ENGINE_HPP
