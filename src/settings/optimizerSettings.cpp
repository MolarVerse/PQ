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
std::string settings::string(const LREnum method)
{
    switch (method)
    {
        case LREnum::CONSTANT: return "CONSTANT";

        case LREnum::CONSTANT_DECAY: return "CONSTANT-DECAY";

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
        setLearningRateStrategy(LREnum::CONSTANT);

    else if ("constant-decay" == method)
        setLearningRateStrategy(LREnum::CONSTANT_DECAY);

    else
        setLearningRateStrategy(LREnum::NONE);
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param method
 */
void OptimizerSettings::setLearningRateStrategy(const LREnum method)
{
    _LRStrategy = method;
}

/**
 * @brief sets the number of epochs
 *
 * @param nEpochs
 */
void OptimizerSettings::setNumberOfEpochs(const size_t nEpochs)
{
    _nEpochs = nEpochs;
}

/**
 * @brief sets the learning rate update frequency
 *
 * @param frequency
 */
void OptimizerSettings::setLRUpdateFrequency(const size_t frequency)
{
    _LRupdateFrequency = frequency;
}

/**
 * @brief sets the initial learning rate
 *
 * @param learningRate
 */
void OptimizerSettings::setInitialLearningRate(const double learningRate)
{
    _initialLearningRate = learningRate;
}

/**
 * @brief sets the learning rate decay
 *
 * @param decay
 */
void OptimizerSettings::setLearningRateDecay(const double decay)
{
    _learningRateDecay = decay;
}

/**
 * @brief sets the min learning rate
 *
 * @param minLearningRate
 */
void OptimizerSettings::setMinLearningRate(const double minLearningRate)
{
    _minLearningRate = minLearningRate;
}

/**
 * @brief sets the max learning rate
 *
 * @param maxLearningRate
 */
void OptimizerSettings::setMaxLearningRate(const double maxLearningRate)
{
    _maxLearningRate = maxLearningRate;
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
settings::Optimizer OptimizerSettings::getOptimizer() { return _optimizer; }

/**
 * @brief returns the learning rate strategy as string
 *
 * @return LearningRateStrategy
 */
LREnum OptimizerSettings::getLearningRateStrategy() { return _LRStrategy; }

/**
 * @brief returns the number of epochs
 *
 * @return size_t
 */
size_t OptimizerSettings::getNumberOfEpochs() { return _nEpochs; }

/**
 * @brief returns the learning rate update frequency
 *
 * @return size_t
 */
size_t OptimizerSettings::getLRUpdateFrequency() { return _LRupdateFrequency; }

/**
 * @brief returns the initial learning rate
 *
 * @return double
 */
double OptimizerSettings::getInitialLearningRate()
{
    return _initialLearningRate;
}

/**
 * @brief returns the min learning rate
 *
 * @return double
 */
double OptimizerSettings::getMinLearningRate() { return _minLearningRate; }

/**
 * @brief returns the learning rate decay
 *
 * @return std::optional<double>
 */
std::optional<double> OptimizerSettings::getLearningRateDecay()
{
    return _learningRateDecay;
}

/**
 * @brief returns the max learning rate
 *
 * @return std::optional<double>
 */
std::optional<double> OptimizerSettings::getMaxLearningRate()
{
    return _maxLearningRate;
}