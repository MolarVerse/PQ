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

#include "stringUtilities.hpp"   // for toLowerCopy

using namespace settings;
using namespace utilities;

/**
 * @brief returns the optimizer as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const OptimizerType method)
{
    switch (method)
    {
        using enum OptimizerType;

        case STEEPEST_DESCENT: return "STEEPEST-DESCENT";
        case ADAM: return "ADAM";

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
        using enum LREnum;

        case CONSTANT: return "CONSTANT";
        case CONSTANT_DECAY: return "CONSTANT-DECAY";
        case EXPONENTIAL_DECAY: return "EXPONENTIAL-DECAY";
        case LINESEARCH_WOLFE: return "LINESEARCH-WOLFE";

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
    using enum OptimizerType;
    const auto optimizerLower = toLowerCopy(optimizer);

    if ("steepest-descent" == optimizerLower)
        setOptimizer(OptimizerType::STEEPEST_DESCENT);

    else if ("adam" == optimizerLower)
        setOptimizer(OptimizerType::ADAM);

    else
        setOptimizer(OptimizerType::NONE);
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param optimizer
 */
void OptimizerSettings::setOptimizer(const OptimizerType optimizer)
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
    using enum LREnum;

    const auto methodLower = toLowerCopy(method);

    if ("constant" == methodLower)
        setLearningRateStrategy(CONSTANT);

    else if ("constant-decay" == methodLower)
        setLearningRateStrategy(CONSTANT_DECAY);

    else if ("exponential-decay" == methodLower)
        setLearningRateStrategy(EXPONENTIAL_DECAY);

    else if ("linesearch-wolfe" == methodLower)
        setLearningRateStrategy(LINESEARCH_WOLFE);

    else
        setLearningRateStrategy(NONE);
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
 * @return OptimizerType
 */
settings::OptimizerType OptimizerSettings::getOptimizer() { return _optimizer; }

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