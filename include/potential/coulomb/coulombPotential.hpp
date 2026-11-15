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

#ifndef POTENTIAL__COULOMB__COULOMB_POTENTIAL_HPP
#define POTENTIAL__COULOMB__COULOMB_POTENTIAL_HPP

#include <mstd/utils/members.hpp>
#include <utility>

#include "unit_types.hpp"

namespace pot
{
    /**
     * @class CoulombPotential
     *
     * @brief CoulombPotential is a base class for all coulomb type potentials
     *
     * @details it inherits in general the coulombRadiusCutOff,
     * coulombEnergyCutOff and coulombForceCutOff to all coulomb type potentials
     *
     * @details available for:
     * - shifted potential
     * - wolf summation as long range correction
     *
     */
    class CoulombPotential
    {
       protected:
        inline static Distance           _coulombRadiusCutOff;
        inline static InvDistance        _coulombEnergyCutOff;
        inline static InvDistanceSquared _coulombForceCutOff;
        inline static InvDistanceCubed   _coulombCutOffCubedInverse;

       public:
        virtual ~CoulombPotential() = default;
        explicit CoulombPotential(const Distance& coulombRadiusCutOff);

        [[nodiscard]]
        virtual std::pair<EnergyPerMol, Force> calculate(
            const Distance&      distance,
            const ChargeSquared& chargeProduct
        ) const = 0;

        STATIC_GETTER(_coulombRadiusCutOff)
        STATIC_GETTER_SETTER(_coulombEnergyCutOff)
        STATIC_GETTER_SETTER(_coulombForceCutOff)
        STATIC_GETTER(_coulombCutOffCubedInverse)

        static void setCoulombRadiusCutOff(const Distance& coulombRadiusCutOff);
    };

}   // namespace pot

#endif   // POTENTIAL__COULOMB__COULOMB_POTENTIAL_HPP
