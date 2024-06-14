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

#include "optimizerSetup.hpp"

#include <memory>

#include "constantStrategy.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizerSettings.hpp"
#include "settings.hpp"
#include "steepestDescent.hpp"

using setup::OptimizerSetup;

/**
 * @brief Wrapper for the optimizer setup
 *
 * @param engine
 */
void setup::setupOptimizer(engine::Engine &engine)
{
    if (!settings::Settings::isOptJobType())
        return;

    engine.getStdoutOutput().writeSetup("optimizer");
    engine.getLogOutput().writeSetup("optimizer");

    OptimizerSetup optimizerSetup(dynamic_cast<engine::OptEngine &>(engine));
    optimizerSetup.setup();
}

/**
 * @brief Construct a new OptimizerSetup object
 *
 * @param optEngine
 */
OptimizerSetup::OptimizerSetup(engine::OptEngine &optEngine)
    : _optEngine(optEngine)
{
}

/**
 * @brief Setup the optimizer
 *
 */
void OptimizerSetup::setup()
{
    const auto learningRateStrategy = setupLearningRateStrategy();
    const auto optimizer            = setupEmptyOptimizer();
    const auto evaluator            = setupEvaluator();

    _optEngine.setLearningRateStrategy(learningRateStrategy);
    _optEngine.setOptimizer(optimizer);
    _optEngine.setEvaluator(evaluator);
}

/**
 * @brief Setup an empty optimizer
 *
 */
std::shared_ptr<opt::Optimizer> OptimizerSetup::setupEmptyOptimizer()
{
    const auto nEpochs = settings::OptimizerSettings::getNumberOfEpochs();

    switch (settings::OptimizerSettings::getOptimizer())
    {
        case settings::Optimizer::STEEPEST_DESCENT:
        {
            return std::make_shared<opt::SteepestDescent>(nEpochs);
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown optimizer type {}",
                string(settings::OptimizerSettings::getOptimizer())
            ));
        }
    }
}

/**
 * @brief Setup the learning rate strategy
 *
 */
std::shared_ptr<opt::LearningRateStrategy> OptimizerSetup::
    setupLearningRateStrategy()
{
    const auto alpha_0 = settings::OptimizerSettings::getInitialLearningRate();

    switch (settings::OptimizerSettings::getLearningRateStrategy())
    {
        case settings::LearningRateStrategy::CONSTANT:
        {
            return std::make_shared<opt::ConstantLRStrategy>(alpha_0);
        }
        case settings::LearningRateStrategy::DECAY:
        {
            throw customException::UserInputException(
                "Decay learning rate strategy not implemented yet"
            );
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown learning rate strategy type {}",
                string(settings::OptimizerSettings::getLearningRateStrategy())
            ));
        }
    }
}

/**
 * @brief Setup the evaluator
 *
 */
std::shared_ptr<opt::Evaluator> OptimizerSetup::setupEvaluator()
{
    if (settings::Settings::getJobtype() == settings::JobType::MM_OPT)
        return std::make_shared<opt::MMEvaluator>();

    else
        throw customException::UserInputException(
            "Unknown job type for the optimizer in order to setup up the "
            "evaluator"
        );
}