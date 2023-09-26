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

#ifndef _COULOMB_WOLF_HPP_

#define _COULOMB_WOLF_HPP_

#include "coulombPotential.hpp"

#include <utility>   // for pair

namespace potential
{
    /**
     * @class CoulombWolf
     *
     * @brief
     * CoulombWolf inherits CoulombPotential
     * CoulombWolf is a class for the Coulomb potential with Wolf summation as long range correction
     *
     */
    class CoulombWolf : public CoulombPotential
    {
      protected:
        inline static double _kappa;
        inline static double _wolfParameter1;
        inline static double _wolfParameter2;
        inline static double _wolfParameter3;

      public:
        explicit CoulombWolf(const double coulombRadiusCutOff, const double kappa);

        [[nodiscard]] std::pair<double, double> calculate(const double, const double) const override;

        static void setKappa(const double kappa) { _kappa = kappa; }
        static void setWolfParameter1(const double wolfParameter1) { _wolfParameter1 = wolfParameter1; }
        static void setWolfParameter2(const double wolfParameter2) { _wolfParameter2 = wolfParameter2; }
        static void setWolfParameter3(const double wolfParameter3) { _wolfParameter3 = wolfParameter3; }

        [[nodiscard]] double getKappa() const { return _kappa; }
        [[nodiscard]] double getWolfParameter1() const { return _wolfParameter1; }
        [[nodiscard]] double getWolfParameter2() const { return _wolfParameter2; }
        [[nodiscard]] double getWolfParameter3() const { return _wolfParameter3; }
    };

}   // namespace potential

#endif   // _COULOMB_WOLF_HPP_
