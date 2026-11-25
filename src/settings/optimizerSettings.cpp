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

/**
 * @brief validates the selected learning-rate strategy
 */
void OptimizerSettings::validateLearningRateStrategy() const
{
    const auto strategy = _lRStrategy.get();

    if (strategy == LearningRate::LINESEARCH_WOLFE)
    {
        throw exc::UserInputException(
            "The Wolfe line search learning rate strategy is not yet "
            "implemented"
        );
    }

    const auto needsDecay = strategy == LearningRate::CONSTANT_DECAY ||
                            strategy == LearningRate::EXPONENTIAL_DECAY;

    if (needsDecay && !isLearningRateDecaySetByUser())
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
    const auto minLR = _minLearningRate.get();
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
    if (!isMaxLearningRateSetByUser())
        return std::nullopt;

    return _maxLearningRate.get();
}

/**
 * @brief returns whether the learning rate decay has been set
 *
 * @return true if learning rate decay is set, false otherwise
 */
bool OptimizerSettings::isLearningRateDecaySetByUser() const
{
    return _learningRateDecay.isSetByUser();
}

/**
 * @brief returns whether the max learning rate has been set
 *
 * @return true if max learning rate is set, false otherwise
 */
bool OptimizerSettings::isMaxLearningRateSetByUser() const
{
    return _maxLearningRate.isSetByUser();
}
