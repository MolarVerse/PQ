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

#ifndef _BUCKINGHAM_PAIR_HPP_

#define _BUCKINGHAM_PAIR_HPP_

#include <cstddef>   // size_t
#include <utility>   // pair

#include "nonCoulombPair.hpp"

namespace potential
{
    /**
     * @class BuckinghamPair
     *
     * @brief inherits from NonCoulombPair represents a pair of Buckingham types
     *
     */
    class BuckinghamPair : public NonCoulombPair
    {
       private:
        double _a;
        double _dRho;
        double _c6;

       public:
        explicit BuckinghamPair(
            const size_t vanDerWaalsType1,
            const size_t vanDerWaalsType2,
            const double cutOff,
            const double a,
            const double dRho,
            const double c6
        )
            : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
              _a(a),
              _dRho(dRho),
              _c6(c6){};

        explicit BuckinghamPair(
            const double cutOff,
            const double a,
            const double dRho,
            const double c6
        )
            : NonCoulombPair(cutOff), _a(a), _dRho(dRho), _c6(c6){};

        explicit BuckinghamPair(
            const double cutOff,
            const double energyCutoff,
            const double forceCutoff,
            const double a,
            const double dRho,
            const double c6
        )
            : NonCoulombPair(cutOff, energyCutoff, forceCutoff),
              _a(a),
              _dRho(dRho),
              _c6(c6){};

        [[nodiscard]] bool operator==(const BuckinghamPair &other) const;

        [[nodiscard]] std::pair<double, double> calculate(const double distance
        ) const override;

        [[nodiscard]] double getA() const { return _a; }
        [[nodiscard]] double getDRho() const { return _dRho; }
        [[nodiscard]] double getC6() const { return _c6; }
    };

}   // namespace potential

#endif   // _BUCKINGHAM_PAIR_HPP_