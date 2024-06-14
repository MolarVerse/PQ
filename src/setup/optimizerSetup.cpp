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
    auto optimizer = setupEmptyOptimizer();

    _optEngine.setOptimizer(optimizer);
}

/**
 * @brief Setup an empty optimizer
 *
 */
std::shared_ptr<optimization::Optimizer> OptimizerSetup::setupEmptyOptimizer()
{
    const auto nEpochs = settings::OptimizerSettings::getNumberOfEpochs();

    switch (settings::OptimizerSettings::getOptimizer())
    {
        case settings::Optimizer::STEEPEST_DESCENT:
        {
            return std::make_shared<optimization::SteepestDescent>(nEpochs);
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
std::shared_ptr<optimization::LearningRateStrategy> OptimizerSetup::
    setupLearningRateStrategy()
{
    const auto alpha_0 = settings::OptimizerSettings::getInitialLearningRate();

    switch (settings::OptimizerSettings::getLearningRateStrategy())
    {
        case settings::LearningRateStrategy::CONSTANT:
        {
            break;
        }
        case settings::LearningRateStrategy::DECAY:
        {
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