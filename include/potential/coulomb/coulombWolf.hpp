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

#include <utility>

#include "coulombPotential.hpp"

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
        static inline double _kappa;
        static inline double _wolfParam1;
        static inline double _wolfParam2;
        static inline double _wolfParam3;

       public:
        explicit CoulombWolf(
            const PotentialSettings& potentialSettings,
            double                   kappa
        );

        [[nodiscard]]
        std::pair<double, double> calculate(
            double distance,
            double chargeProduct
        ) const override;
    };

}   // namespace pot

#endif   // POTENTIAL__COULOMB__COULOMB_WOLF_HPP
