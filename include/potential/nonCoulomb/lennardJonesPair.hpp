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

#ifndef POTENTIAL__NON_COULOMB__LENNARD_JONES_PAIR_HPP
#define POTENTIAL__NON_COULOMB__LENNARD_JONES_PAIR_HPP

#include <utility>

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
            ExtVdwType      vanDerWaalsType1,
            ExtVdwType      vanDerWaalsType2,
            const Pos      &cutOff,
            const LJParams &params
        );

        explicit LennardJonesPair(const Pos &cutOff, const LJParams &params);

        explicit LennardJonesPair(
            const Pos          &cutOff,
            const EnergyPerMol &energyCutoff,
            const Force        &forceCutoff,
            const LJParams     &params
        );

        [[nodiscard]]
        bool operator==(const LennardJonesPair &other) const;

        [[nodiscard]]
        std::pair<EnergyPerMol, Force> calculate(
            const Pos &distance
        ) const override;

        friend struct ::TestLJPairUtils;
    };

}   // namespace pot

#endif   // POTENTIAL__NON_COULOMB__LENNARD_JONES_PAIR_HPP
