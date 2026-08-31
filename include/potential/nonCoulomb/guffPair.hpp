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

#ifndef _GUFF_PAIR_HPP_

#define _GUFF_PAIR_HPP_

#include <utility>   // pair

#include "defaults.hpp"
#include "nonCoulombPair.hpp"

struct TestGuffPairUtils;   // forward declaration

namespace potential
{
    /**
     * @class GuffPair
     *
     * @brief inherits from NonCoulombPair represents a pair of Guff types (full
     * guff formula)
     *
     * @note here the constructor including the van der Waals types is missing
     * as this class is only used for guff potentials. Therefore also the
     * comparison operator == is missing.
     *
     */
    class GuffPair : public NonCoulombPair
    {
       private:
        std::array<double, defaults::NUM_GUFF_COEFFICIENTS> _coefficients{};

       public:
        explicit GuffPair(
            const double,
            const std::array<double, defaults::NUM_GUFF_COEFFICIENTS>&
        );
        explicit GuffPair(
            const double,
            const double,
            const double,
            const std::array<double, defaults::NUM_GUFF_COEFFICIENTS>&
        );

        [[nodiscard]]
        std::pair<double, double> calculate(
            const double distance
        ) const override;

        friend struct ::TestGuffPairUtils;
    };

}   // namespace potential

#endif   // _GUFF_PAIR_HPP_
