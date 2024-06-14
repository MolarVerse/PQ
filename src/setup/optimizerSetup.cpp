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

#include "optEngine.hpp"
#include "optimizerSettings.hpp"
#include "settings.hpp"
#include "steepestDescent.hpp"

using setup::OptimizerSetup;
using namespace settings;

/**
 * @brief Wrapper for the optimizer setup
 *
 * @param engine
 */
void setup::setupOptimizer(engine::Engine &engine)
{
    if (!Settings::isOptJobType())
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
    const auto nEpochs        = OptimizerSettings::getNumberOfEpochs();
    const auto learningRate_0 = OptimizerSettings::getInitialLearningRate();

    switch (OptimizerSettings::getOptimizer())
    {
        case Optimizer::STEEPEST_DESCENT:
        {
            _optEngine.makeOptimizer(
                optimizer::SteepestDescent(nEpochs, learningRate_0)
            );
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown optimizer type {}",
                string(OptimizerSettings::getOptimizer())
            ));
        }
    }
}