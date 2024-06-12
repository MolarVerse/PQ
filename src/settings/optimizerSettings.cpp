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

#include "optimizerSettings.hpp"

using namespace settings;

/**
 * @brief returns the optimizer as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const Optimizer method)
{
    switch (method)
    {
        case Optimizer::STEEPEST_DESCENT: return "STEEPEST-DESCENT";

        default: return "none";
    }
}

/**
 * @brief returns the learning rate strategy as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const LearningRateStrategy method)
{
    switch (method)
    {
        case LearningRateStrategy::CONSTANT: return "CONSTANT";
        case LearningRateStrategy::DECAY: return "DECAY";

        default: return "none";
    }
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param optimizer
 */
void OptimizerSettings::setOptimizer(const std::string_view &optimizer)
{
    if ("steepest-descent" == optimizer)
        setOptimizer(Optimizer::STEEPEST_DESCENT);
    else
        setOptimizer(Optimizer::NONE);
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param optimizer
 */
void OptimizerSettings::setOptimizer(const Optimizer optimizer)
{
    _optimizer = optimizer;
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param method
 */
void OptimizerSettings::setLearningRateStrategy(const std::string_view &method)
{
    if ("constant" == method)
        setLearningRateStrategy(LearningRateStrategy::CONSTANT);
    else if ("decay" == method)
        setLearningRateStrategy(LearningRateStrategy::DECAY);
    else
        setLearningRateStrategy(LearningRateStrategy::NONE);
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param method
 */
void OptimizerSettings::setLearningRateStrategy(
    const LearningRateStrategy method
)
{
    _learningRateStrategy = method;
}

/**
 * @brief sets the initial learning rate
 *
 * @param learningRate
 */
void OptimizerSettings::setInitialLearningRate(const double learningRate)
{
    _initialLearningRate      = learningRate;
    _isInitialLearningRateSet = true;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief returns the optimizer as string
 *
 * @return Optimizer
 */
Optimizer OptimizerSettings::getOptimizer() { return _optimizer; }

/**
 * @brief returns the learning rate strategy as string
 *
 * @return LearningRateStrategy
 */
LearningRateStrategy OptimizerSettings::getLearningRateStrategy()
{
    return _learningRateStrategy;
}

/**
 * @brief returns the initial learning rate
 *
 * @return double
 */
double OptimizerSettings::getInitialLearningRate()
{
    return _initialLearningRate;
}