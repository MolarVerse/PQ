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

#include <utility>   // pair

#include "nonCoulombPair.hpp"
#include "strongTypes.hpp"

struct TestBuckinghamPairUtils;   // forward declaration

namespace pot
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
        BuckinghamParams _params;

       public:
        explicit BuckinghamPair(
            const ExtVdwType        vanDerWaalsType1,
            const ExtVdwType        vanDerWaalsType2,
            const double            cutOff,
            const BuckinghamParams& params
        );

        explicit BuckinghamPair(
            const double            cutOff,
            const BuckinghamParams& params
        );

        explicit BuckinghamPair(
            const double            cutOff,
            const double            energyCutoff,
            const double            forceCutoff,
            const BuckinghamParams& params
        );

        // TODO: we need to explicitly delete it to not implicitly create it
        // with the wrong types!!! Needs cleanup
        explicit BuckinghamPair(
            const size_t,
            const size_t,
            const double,
            const BuckinghamParams& params
        ) = delete;

        [[nodiscard]] bool operator==(const BuckinghamPair& other) const;

        [[nodiscard]] std::pair<double, double> calculate(
            const double distance
        ) const override;

        friend struct ::TestBuckinghamPairUtils;
    };

}   // namespace pot

#endif   // _BUCKINGHAM_PAIR_HPP_
