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

#ifndef _MORSE_PAIR_HPP_

#define _MORSE_PAIR_HPP_

#include <utility>   // pair

#include "nonCoulombPair.hpp"
#include "strongTypes.hpp"

struct TestMorsePairUtils;   // forward declaration

namespace potential
{
    /**
     * @class MorsePair
     *
     * @brief inherits from NonCoulombPair represents a pair of Morse types
     *
     */
    class MorsePair : public NonCoulombPair
    {
       private:
        MorseParams _params;

       public:
        explicit MorsePair(
            const ExtVdwType   vanDerWaalsType1,
            const ExtVdwType   vanDerWaalsType2,
            const double       cutOff,
            const MorseParams &params
        );

        explicit MorsePair(const double cutOff, const MorseParams &params);

        explicit MorsePair(
            const double       cutOff,
            const double       energyCutoff,
            const double       forceCutoff,
            const MorseParams &params
        );

        // TODO: we need to explicitly delete it to not implicitly create it
        // with the wrong types!!! Needs cleanup
        explicit MorsePair(
            const size_t,
            const size_t,
            const double,
            const MorseParams &
        ) = delete;

        [[nodiscard]] bool operator==(const MorsePair &other) const;

        [[nodiscard]] std::pair<double, double> calculate(
            const double distance
        ) const override;

        friend struct ::TestMorsePairUtils;
    };

}   // namespace potential

#endif   // _MORSE_PAIR_HPP_
