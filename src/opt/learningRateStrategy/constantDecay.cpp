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

#include "constantDecay.hpp"

using namespace opt;

/**
 * @brief Construct a new Constant Decay L R Strategy:: Constant Decay L R
 * Strategy object
 *
 * @param initialLearningRate
 * @param decay
 * @param frequency
 */
ConstantDecayLRStrategy::ConstantDecayLRStrategy(
    const double initialLearningRate,
    const double decay,
    const size_t frequency
)
    : LearningRateStrategy(initialLearningRate, frequency), _decay(decay)
{
}

/**
 * @brief Construct a new Constant Decay L R Strategy:: Constant Decay L R
 * Strategy object
 *
 * @param initialLearningRate
 * @param decay
 */
ConstantDecayLRStrategy::ConstantDecayLRStrategy(
    const double initialLearningRate,
    const double decay
)
    : LearningRateStrategy(initialLearningRate), _decay(decay)
{
}

/**
 * @brief Update the learning rate
 */
void ConstantDecayLRStrategy::updateLearningRate(std::string &message)
{
    if (_counter % _frequency == 0)
        _learningRate -= _decay;

    ++_counter;

    if (_learningRate <= 0.0)
        message = "Learning rate is zero or negative.";
}