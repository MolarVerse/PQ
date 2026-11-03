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

#include <format>

#include "defaults.hpp"
#include "exceptions.hpp"

/**
 * @brief Constructs an OptimizerSettings object with default values
 *
 * @details Initializes the optimizer to STEEPEST_DESCENT and sets all other
 *          static inline settings to their respective default values.
 */
OptimizerSettings::OptimizerSettings()
    : _optimizer(OptimizerType::STEEPEST_DESCENT),
      _lRStrategy(LearningRate::EXPONENTIAL_DECAY),
      _nEpochs(defaults::N_EPOCHS_DEFAULT),
      _lRupdateFrequency(defaults::LR_UPDATE_FREQUENCY_DEFAULT),
      _initialLearningRate(defaults::INITIAL_LEARNING_RATE_DEFAULT),
      _minLearningRate(defaults::MIN_LEARNING_RATE_DEFAULT)
{
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
void OptimizerSettings::setOptimizer(OptimizerType optimizer)
{
    _optimizer.set(optimizer);
}

/**
 * @brief sets the optimizer to enum in settings
 *
 * @param method
 */
void OptimizerSettings::setLearningRateStrategy(LearningRate method)
{
    _lRStrategy.set(method);
}

/**
 * @brief sets the number of epochs
 *
 * @param nEpochs
 */
void OptimizerSettings::setNumberOfEpochs(size_t nEpochs)
{
    _nEpochs.set(nEpochs);
}

/**
 * @brief sets the learning rate update frequency
 *
 * @param frequency
 */
void OptimizerSettings::setLRUpdateFrequency(size_t frequency)
{
    _lRupdateFrequency.set(frequency);
}

/**
 * @brief sets the initial learning rate
 *
 * @param learningRate
 */
void OptimizerSettings::setInitialLearningRate(double learningRate)
{
    _initialLearningRate.set(learningRate);
}

/**
 * @brief sets the learning rate decay
 *
 * @param decay
 */
void OptimizerSettings::setLearningRateDecay(double decay)
{
    _learningRateDecay.set(decay);
}

/**
 * @brief sets the min learning rate
 *
 * @param minLearningRate
 */
void OptimizerSettings::setMinLearningRate(double minLearningRate)
{
    _minLearningRate.set(minLearningRate);
}

/**
 * @brief sets the max learning rate
 *
 * @param maxLearningRate
 */
void OptimizerSettings::setMaxLearningRate(double maxLearningRate)
{
    _maxLearningRate.set(maxLearningRate);
}

/*****************************
 *                           *
 * validation helper methods *
 *                           *
 *****************************/

/**
 * @brief validates the selected learning-rate strategy
 */
void OptimizerSettings::validateLearningRateStrategy() const
{
    const auto strategy = getLearningRateStrategy();

    if (strategy == LearningRate::LINESEARCH_WOLFE)
    {
        throw exc::UserInputException(
            "The Wolfe line search learning rate strategy is not yet "
            "implemented"
        );
    }

    const auto needsDecay = strategy == LearningRate::CONSTANT_DECAY ||
                            strategy == LearningRate::EXPONENTIAL_DECAY;

    if (needsDecay && !isLearningRateDecaySet())
    {
        throw exc::UserInputException(
            std::format(
                "The {} learning rate strategy requires "
                "learning-rate-decay.",
                strategy == LearningRate::CONSTANT_DECAY ? "constant-decay"
                                                         : "exponential-decay"
            )
        );
    }
}

/**
 * @brief validates the configured learning-rate bounds
 */
void OptimizerSettings::validateLearningRateBounds() const
{
    const auto minLR = getMinLearningRate();
    const auto maxLR = getMaxLearningRate();

    if (maxLR.has_value() && minLR >= maxLR.value())
    {
        throw exc::UserInputException(
            std::format(
                "The minimum learning rate {} is greater or equal to the "
                "maximum learning rate {}, which is not allowed.",
                minLR,
                maxLR.value()
            )
        );
    }
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
OptimizerType OptimizerSettings::getOptimizer() const
{
    return _optimizer.get();
}

/**
 * @brief returns the learning rate strategy as string
 *
 * @return LearningRateStrategy
 */
LearningRate OptimizerSettings::getLearningRateStrategy() const
{
    return _lRStrategy.get();
}

/**
 * @brief returns the number of epochs
 *
 * @return size_t
 */
size_t OptimizerSettings::getNumberOfEpochs() const { return _nEpochs.get(); }

/**
 * @brief returns the learning rate update frequency
 *
 * @return size_t
 */
size_t OptimizerSettings::getLRUpdateFrequency() const
{
    return _lRupdateFrequency.get();
}

/**
 * @brief returns the initial learning rate
 *
 * @return double
 */
double OptimizerSettings::getInitialLearningRate() const
{
    return _initialLearningRate.get();
}

/**
 * @brief returns the min learning rate
 *
 * @return double
 */
double OptimizerSettings::getMinLearningRate() const
{
    return _minLearningRate.get();
}

/**
 * @brief returns the learning rate decay
 *
 * @return double
 */
double OptimizerSettings::getLearningRateDecay() const
{
    return _learningRateDecay.get();
}

/**
 * @brief returns the max learning rate
 *
 * @details the maximum is optional and has no default: no maximum is applied
 * when it was not set
 *
 * @return std::optional<double> the max learning rate, std::nullopt if not set
 */
std::optional<double> OptimizerSettings::getMaxLearningRate() const
{
    if (!isMaxLearningRateSet())
        return std::nullopt;

    return _maxLearningRate.get();
}

/**
 * @brief returns whether the learning rate decay has been set
 *
 * @return true if learning rate decay is set, false otherwise
 */
bool OptimizerSettings::isLearningRateDecaySet() const
{
    return _learningRateDecay.isSet();
}

/**
 * @brief returns whether the max learning rate has been set
 *
 * @return true if max learning rate is set, false otherwise
 */
bool OptimizerSettings::isMaxLearningRateSet() const
{
    return _maxLearningRate.isSet();
}
