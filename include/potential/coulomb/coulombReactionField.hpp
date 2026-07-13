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

#ifndef _COULOMB_REACTION_FIELD_HPP_

#define _COULOMB_REACTION_FIELD_HPP_

#include <utility>   // for pair

#include "coulombPotential.hpp"

namespace potential
{
    /**
     * @class CoulombReactionField
     *
     * @brief
     * CoulombReactionField inherits CoulombPotential
     * CoulombReactionField is a class for the reaction field Coulomb
     * potential
     *
     */
    class CoulombReactionField : public CoulombPotential
    {
       private:
        double _epsilon{};       // double check unit!!!
        double _rfPrefactor{};   // double check unit!!!

       public:
        CoulombReactionField(
            const double coulombRadiusCutOff,
            const double epsilon
        )
            : CoulombPotential{coulombRadiusCutOff}, _epsilon{epsilon}
        {
            _rfPrefactor = (epsilon - 1) / ((2 * epsilon) + 1);
        }

        [[nodiscard]] std::pair<double, double> calculate(
            const double distance,
            const double chargeProduct
        ) const override;
    };

}   // namespace potential

#endif   // _COULOMB_REACTION_FIELD_HPP_