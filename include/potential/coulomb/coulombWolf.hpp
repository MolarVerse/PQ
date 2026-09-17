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

#ifndef _COULOMB_WOLF_HPP_

#define _COULOMB_WOLF_HPP_

#include <utility>   // for pair

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
        explicit CoulombWolf(double coulombRadiusCutOff, double kappa);

        [[nodiscard]]
        std::pair<double, double> calculate(
            double distance,
            double chargeProduct
        ) const override;

        /***************************
         * standard setter methods *
         ***************************/

        static void setKappa(double kappa);
        static void setWolfParameter1(double wolfParameter1);
        static void setWolfParameter2(double wolfParameter2);
        static void setWolfParameter3(double wolfParameter3);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static double getKappa();
        [[nodiscard]] static double getWolfParameter1();
        [[nodiscard]] static double getWolfParameter2();
        [[nodiscard]] static double getWolfParameter3();
    };

}   // namespace pot

#endif   // _COULOMB_WOLF_HPP_
