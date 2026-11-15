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

#ifndef POTENTIAL__COULOMB__COULOMB_WOLF_HPP
#define POTENTIAL__COULOMB__COULOMB_WOLF_HPP

#include <mstd/utils/members.hpp>
#include <utility>

#include "coulombPotential.hpp"
#include "unit_types.hpp"

namespace pot
{
    /**
     * @class CoulombWolf
     *
     * @brief
     * CoulombWolf inherits CoulombPotential
     * CoulombWolf is a class for the Coulomb potential with Wolf summation as
     * long range correction
     *
     */
    class CoulombWolf : public CoulombPotential
    {
       protected:
        static inline InvDistance        _kappa;
        static inline InvDistance        _wolfParam1;
        static inline InvDistance        _wolfParam2;
        static inline InvDistanceSquared _wolfParam3;

       public:
        explicit CoulombWolf(
            const Distance&    coulombRadiusCutOff,
            const InvDistance& kappa
        );

        [[nodiscard]]
        std::pair<EnergyPerMol, Force> calculate(
            const Distance&      distance,
            const ChargeSquared& chargeProduct
        ) const override;

        CONST_GETTER_SETTER(_kappa);
        CONST_GETTER_SETTER(_wolfParam1);
        CONST_GETTER_SETTER(_wolfParam2);
        CONST_GETTER_SETTER(_wolfParam3);
    };

}   // namespace pot

#endif   // POTENTIAL__COULOMB__COULOMB_WOLF_HPP
