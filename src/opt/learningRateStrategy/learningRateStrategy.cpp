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

#include "learningRateStrategy.hpp"

#include <format>   // for format

using namespace opt;

/**
 * @brief Construct a new LearningRateStrategy object
 *
 * @param initialLearningRate
 */
LearningRateStrategy::LearningRateStrategy(const double initialLearningRate)
    : _initialLearningRate(initialLearningRate),
      _learningRate(initialLearningRate)
{
}

/**
 * @brief Construct a new LearningRateStrategy object
 *
 * @param initialLearningRate
 * @param frequency
 */
LearningRateStrategy::LearningRateStrategy(
    const double initialLearningRate,
    const size_t frequency
)
    : _frequency(frequency),
      _initialLearningRate(initialLearningRate),
      _learningRate(initialLearningRate)
{
}

/**
 * @brief check the learning rate if it is within the bounds
 *       and update the learning rate
 */
void LearningRateStrategy::checkLearningRate()
{
    if (_maxLearningRate.has_value())
        if (_learningRate > _maxLearningRate.value())
        {
            _learningRate      = _maxLearningRate.value();
            const auto message = std::format(
                "Learning rate {} is greater than the maximum learning rate "
                "{}. Therefore, the learning rate is set to the maximum "
                "learning rate.",
                _learningRate,
                _maxLearningRate.value()
            );

            _warningMessages.push_back(message);
        }

    if (_learningRate < _minLearningRate)
    {
        _learningRate      = _minLearningRate;
        const auto message = std::format(
            "Learning rate {} is less than the minimum learning rate {}. "
            "Therefore, the learning rate is set to the minimum learning "
            "rate.",
            _learningRate,
            _minLearningRate
        );

        _warningMessages.push_back(message);
    }
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Get the learning rate
 *
 * @return double
 */
double LearningRateStrategy::getLearningRate() const { return _learningRate; }

/**
 * @brief Get the warning messages
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> LearningRateStrategy::getWarningMessages() const
{
    return _warningMessages;
}

/**
 * @brief Get the error messages
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> LearningRateStrategy::getErrorMessages() const
{
    return _errorMessages;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the evaluator from a shared pointer
 *
 */
void LearningRateStrategy::setEvaluator(
    const std::shared_ptr<Evaluator> evaluator
)
{
    _evaluator = evaluator;
}

/**
 * @brief set the optimizer from a shared pointer
 *
 * @param optimizer - std::shared_ptr<Optimizer>
 */
void LearningRateStrategy::setOptimizer(
    const std::shared_ptr<Optimizer> optimizer
)
{
    _optimizer = optimizer;
}

/**
 * @brief set the minimum learning rate
 *
 * @param minLearningRate - double
 */
void LearningRateStrategy::setMinLearningRate(const double minLearningRate)
{
    _minLearningRate = minLearningRate;
}

/**
 * @brief set the maximum learning rate
 *
 * @param maxLearningRate - std::optional<double>
 */
void LearningRateStrategy::setMaxLearningRate(
    const std::optional<double> maxLearningRate
)
{
    _maxLearningRate = maxLearningRate;
}