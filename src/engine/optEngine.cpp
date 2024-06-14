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

#include "optEngine.hpp"

using namespace engine;
using namespace opt;

/**
 * @brief set the optimizer from a shared pointer
 *
 * @param optimizer
 */
void OptEngine::setOptimizer(const std::shared_ptr<Optimizer> &optimizer)
{
    _optimizer = optimizer;
}

/**
 * @brief set the learning rate strategy from a shared pointer
 *
 * @param learningRateStrategy
 */
void OptEngine::setLearningRateStrategy(
    const std::shared_ptr<LearningRateStrategy> &strategy
)
{
    _learningRateStrategy = strategy;
}

/**
 * @brief set the evaluator from a shared pointer
 *
 * @param evaluator
 */
void OptEngine::setEvaluator(const std::shared_ptr<Evaluator> &evaluator)
{
    _evaluator = evaluator;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the optimizer
 *
 * @return std::shared_ptr<Optimizer>
 */
std::shared_ptr<Optimizer> &OptEngine::getOptimizer() { return _optimizer; }

/**
 * @brief get the learning rate strategy
 *
 * @return std::shared_ptr<LearningRateStrategy>
 */
std::shared_ptr<LearningRateStrategy> &OptEngine::getLearningRateStrategy()
{
    return _learningRateStrategy;
}

/**
 * @brief get the evaluator
 *
 * @return std::shared_ptr<Evaluator>
 */
std::shared_ptr<Evaluator> &OptEngine::getEvaluator() { return _evaluator; }