/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _COULOMB_SHIFTED_POTENTIAL_HPP_

#define _COULOMB_SHIFTED_POTENTIAL_HPP_

#include "coulombPotential.hpp"

#include <utility>   // for pair

namespace potential
{
    /**
     * @class CoulombShiftedPotential
     *
     * @brief
     * CoulombShiftedPotential inherits CoulombPotential
     * CoulombShiftedPotential is a class for the shifted Coulomb potential
     *
     */
    class CoulombShiftedPotential : public potential::CoulombPotential
    {
      public:
        using CoulombPotential::CoulombPotential;

        [[nodiscard]] std::pair<double, double> calculate(const double, const double) const override;
    };

}   // namespace potential

#endif   // _COULOMB_SHIFTED_POTENTIAL_HPP_