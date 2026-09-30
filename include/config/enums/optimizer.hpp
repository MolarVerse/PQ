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

#ifndef _OPTIMIZER_ENUM_HPP_
#define _OPTIMIZER_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief enum OptimizerType
 *
 */
enum class OptimizerType : std::uint8_t;

#define OPTIMIZER_TYPE_LIST(X) \
    X(STEEPEST_DESCENT)        \
    X(ADAM)

MSTD_ENUM(OptimizerType, std::uint8_t, OPTIMIZER_TYPE_LIST)

#undef OPTIMIZER_TYPE_LIST

/**
 * @brief enum LearningRate
 *
 */
enum class LearningRate : std::uint8_t;

#define LEARNING_RATE_LIST(X) \
    X(CONSTANT)               \
    X(CONSTANT_DECAY)         \
    X(EXPONENTIAL_DECAY)      \
    X(LINESEARCH_WOLFE)

MSTD_ENUM(LearningRate, std::uint8_t, LEARNING_RATE_LIST)

namespace mstd
{

    /**
     * @brief Input alias for LearningRate enum
     *
     * @tparam LearningRate The enum type for which this input alias is defined.
     */
    template <>
    struct EnumAliases<LearningRate>
    {
        static constexpr auto value = makeAliases<LearningRate>(
            {{"linesearch", LearningRate::LINESEARCH_WOLFE}}
        );
    };
}   // namespace mstd

#undef LEARNING_RATE_LIST

#endif   // _OPTIMIZER_ENUM_HPP_
