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

#ifndef _FIRST_ORDER_DECAY_HPP_

#define _FIRST_ORDER_DECAY_HPP_

#include <cmath>   // for exp

#include "learningRateStrategy.hpp"

namespace opt
{
    /**
     * @brief First order decay learning rate strategy
     *
     */
    class FirstOrderDecayLRStrategy : public LearningRateStrategy
    {
       private:
        double _decay;

       public:
        FirstOrderDecayLRStrategy() = default;
        FirstOrderDecayLRStrategy(const double, const double);

        void updateLearningRate() override;
    };

}   // namespace opt

#endif   // _FIRST_ORDER_DECAY_HPP_