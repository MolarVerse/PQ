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

// Fixed-work benchmark of the learning rate strategies: the constant, the
// constant decay and the exponential decay strategy, called through the base
// class as the optimization loop does.

#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <vector>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "constant.hpp"
#include "constantDecay.hpp"
#include "expDecay.hpp"
#include "learningRateStrategy.hpp"
#include "perfAlloc.hpp"

static constexpr std::uint64_t ITERATIONS = 200000;

int main()
{
    std::vector<std::shared_ptr<opt::LearningRateStrategy>> strategies{
        std::make_shared<opt::ConstantLRStrategy>(0.1),
        std::make_shared<opt::ConstantDecayLRStrategy>(0.1, 1.0e-9, 10),
        std::make_shared<opt::ExpDecayLR>(0.1, 2.0, 10)
    };

    perfAllocBegin();
    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
        for (const auto &strategy : strategies)
            strategy->updateLearningRate(i, ITERATIONS);

    auto sum = 0.0;
    for (const auto &strategy : strategies) sum += strategy->getLearningRate();

    std::cout << std::format("{:.9f}\n", sum);
    return 0;
}
