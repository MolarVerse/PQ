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

#include "constant.hpp"

using namespace opt;

/**
 * @brief Construct a new ConstantLRStrategy::ConstantLRStrategy object
 *
 * @param initialLearningRate
 */
ConstantLRStrategy::ConstantLRStrategy(const double initialLearningRate)
    : LearningRateStrategy(initialLearningRate)
{
}

/**
 * @brief Clone the learning rate strategy
 *
 * @return std::shared_ptr<LearningRateStrategy>
 */
std::shared_ptr<LearningRateStrategy> ConstantLRStrategy::clone() const
{
    return std::make_shared<ConstantLRStrategy>(*this);
}

/**
 * @brief Update the learning rate
 *
 * @details This function does nothing, as the learning rate is constant.
 */
void ConstantLRStrategy::updateLearningRate(const size_t, const size_t) {}