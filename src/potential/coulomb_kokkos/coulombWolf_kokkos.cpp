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

#include "coulombWolf_kokkos.hpp"

using namespace potential;

/**
 * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
 *
 * @details this constructor calculates automatically the three need wolf
 * parameters from kappa in order to gain speed
 *
 * @param coulombRadiusCutOff
 * @param kappa
 */
KokkosCoulombWolf::KokkosCoulombWolf(
    const double coulombRadiusCutOff,
    const double kappa,
    const double wolfParameter1,
    const double wolfParameter2,
    const double wolfParameter3,
    const double prefactor
)
    : _coulombRadiusCutOff("coulombRadiusCutOff", 1),
      _kappa("kappa", 1),
      _wolfParameter1("wolfParameter1", 1),
      _wolfParameter2("wolfParameter2", 1),
      _wolfParameter3("wolfParameter3", 1),
      _prefactor("prefactor", 1)

{
    _coulombRadiusCutOff.h_view() = coulombRadiusCutOff;
    _kappa.h_view()               = kappa;
    _wolfParameter1.h_view()      = wolfParameter1;
    _wolfParameter2.h_view()      = wolfParameter2;
    _wolfParameter3.h_view()      = wolfParameter3;
    _prefactor.h_view()           = prefactor;

    Kokkos::deep_copy(_coulombRadiusCutOff.d_view, _coulombRadiusCutOff.h_view);
    Kokkos::deep_copy(_kappa.d_view, _kappa.h_view);
    Kokkos::deep_copy(_wolfParameter1.d_view, _wolfParameter1.h_view);
    Kokkos::deep_copy(_wolfParameter2.d_view, _wolfParameter2.h_view);
    Kokkos::deep_copy(_wolfParameter3.d_view, _wolfParameter3.h_view);
    Kokkos::deep_copy(_prefactor.d_view, _prefactor.h_view);
}

/**
 * @brief calculate the energy and force of the Coulomb potential with Wolf
 * summation as long range correction
 *
 * @link https://doi.org/10.1063/1.478738
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] double KokkosCoulombWolf::calculate(
    const double distance,
    const double charge_i,
    const double charge_j,
    const double dxyz[3],
    double      *force
) const
{
    const auto coulombPrefactor = charge_i * charge_j * _prefactor.d_view();

    const auto kappaDistance = _kappa.d_view() * distance;
    const auto erfcFactor    = Kokkos::erfc(kappaDistance);

    auto energy = erfcFactor / distance - _wolfParameter1.d_view();
    energy +=
        _wolfParameter3.d_view() * (distance - _coulombRadiusCutOff.d_view());

    auto scalarForce  = erfcFactor / (distance * distance);
    scalarForce      += _wolfParameter2.d_view() *
                   Kokkos::exp(-kappaDistance * kappaDistance) / distance;
    scalarForce -= _wolfParameter3.d_view();

    scalarForce *= coulombPrefactor;
    scalarForce /= distance;

    force[0] += scalarForce * dxyz[0];
    force[1] += scalarForce * dxyz[1];
    force[2] += scalarForce * dxyz[2];

    energy *= coulombPrefactor;
    return energy;
}