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

#ifndef _LENNARD_JONES_PAIR_HPP_

#define _LENNARD_JONES_PAIR_HPP_

#include <utility>   // pair

#include "nonCoulombPair.hpp"
#include "strongTypes.hpp"

struct TestLJPairUtils;

namespace pot
{

    /**
     * @class LennardJonesPair
     *
     * @brief inherits from NonCoulombPair and represents a pair of
     * Lennard-Jones types
     *
     */
    class LennardJonesPair : public NonCoulombPair
    {
       private:
        LJParams _params;

       public:
        explicit LennardJonesPair(
            const ExtVdwType vanDerWaalsType1,
            const ExtVdwType vanDerWaalsType2,
            const double     cutOff,
            const LJParams  &params
        );

        explicit LennardJonesPair(const double cutOff, const LJParams &params);

        explicit LennardJonesPair(
            const double    cutOff,
            const double    energyCutoff,
            const double    forceCutoff,
            const LJParams &params
        );

        // TODO: we need to explicitly delete it to not implictly create it with
        // the wrong types!!! Needs cleanup
        explicit LennardJonesPair(
            const size_t,
            const size_t,
            const double,
            const LJParams &
        ) = delete;

        [[nodiscard]]
        bool operator==(const LennardJonesPair &other) const;

        [[nodiscard]] std::pair<double, double> calculate(
            const double distance
        ) const override;

        friend struct ::TestLJPairUtils;
    };

}   // namespace pot

#endif   // _LENNARD_JONES_PAIR_HPP_
