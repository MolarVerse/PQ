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

#ifndef _KOKKOS_COULOMB_WOLF_HPP_

#define _KOKKOS_COULOMB_WOLF_HPP_

#include <Kokkos_DualView.hpp>
#include <utility>   // for pair

namespace potential
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
    class KokkosCoulombWolf
    {
       private:
        Kokkos::DualView<double> _coulombRadiusCutOff;
        Kokkos::DualView<double> _kappa;
        Kokkos::DualView<double> _wolfParameter1;
        Kokkos::DualView<double> _wolfParameter2;
        Kokkos::DualView<double> _wolfParameter3;
        Kokkos::DualView<double> _prefactor;

       public:
        KokkosCoulombWolf(
            const double coulombRadiusCutOff,
            const double kappa,
            const double wolfParameter1,
            const double wolfParameter2,
            const double wolfParameter3,
            const double prefactor
        );

        KokkosCoulombWolf()  = default;
        ~KokkosCoulombWolf() = default;

        KOKKOS_INLINE_FUNCTION double calculate(
            const float distance,
            const float charge_i,
            const float charge_j,
            float      &force
        ) const
        {
            const auto prefactor      = _prefactor.d_view();
            const auto kappa          = _kappa.d_view();
            const auto wolfParameter1 = _wolfParameter1.d_view();
            const auto wolfParameter2 = _wolfParameter2.d_view();
            const auto wolfParameter3 = _wolfParameter3.d_view();
            const auto rcCutOff       = _coulombRadiusCutOff.d_view();

            const auto coulombPrefactor = charge_i * charge_j * prefactor;

            const auto kappaDistance        = kappa * distance;
            const auto kappaDistanceSquared = kappaDistance * kappaDistance;
            const auto erfcFactor           = Kokkos::erfc(kappaDistance);

            auto energy  = erfcFactor / distance - wolfParameter1;
            energy      += wolfParameter3 * (distance - rcCutOff);

            auto scalarForce  = erfcFactor / (distance * distance);
            scalarForce      -= wolfParameter3;
            scalarForce +=
                wolfParameter2 * Kokkos::exp(-kappaDistanceSquared) / distance;

            scalarForce *= coulombPrefactor;

            force += scalarForce;

            energy *= coulombPrefactor;
            return energy;
        }

        Kokkos::View<double> getCoulombRadiusCutOff() const
        {
            return _coulombRadiusCutOff.d_view;
        }
    };

}   // namespace potential

#endif   // _KOKKOS_COULOMB_WOLF_HPP_
