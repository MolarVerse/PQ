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

#ifndef _CONSTANT_DECAY_LEARNING_RATE_STRATEGY_HPP_

#define _CONSTANT_DECAY_LEARNING_RATE_STRATEGY_HPP_

#include "learningRateStrategy.hpp"

namespace opt
{
    /**
     * @class ConstantDecayLRStrategy
     *
     * @brief Constant Decay Learning Rate Strategy
     *
     */
    class ConstantDecayLRStrategy : public LearningRateStrategy
    {
       private:
        double _decay;

       public:
        explicit ConstantDecayLRStrategy(
            const double initialLearningRate,
            const double decay,
            const size_t frequency
        );

        explicit ConstantDecayLRStrategy(
            const double initialLearningRate,
            const double decay
        );

        ConstantDecayLRStrategy()           = default;
        ~ConstantDecayLRStrategy() override = default;

        void updateLearningRate(std::string &message) override;
    };

}   // namespace opt

#endif   // _CONSTANT_DECAY_LEARNING_RATE_STRATEGY_HPP_