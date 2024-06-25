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

#include "expDecay.hpp"

#include "optimizerSettings.hpp"

using namespace opt;
using namespace settings;

/**
 * @brief Construct a new Exp Decay L R Strategy:: Exp Decay L R Strategy object
 *
 * @param initialLearningRate
 * @param decay
 * @param frequency
 */
ExpDecayLR::ExpDecayLR(const double initialLearningRate, const double decay)
    : LearningRateStrategy(initialLearningRate), _decay(decay)
{
}

/**
 * @brief Update the learning rate
 */
void ExpDecayLR::updateLearningRate(const size_t step, const size_t nEpochs)
{
    const auto factor = std::exp(-_decay * double(step) / double(nEpochs));

    _learningRate = _initialLearningRate * factor;
}