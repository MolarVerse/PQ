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
using namespace Kokkos;

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
      _wolfParam1("wolfParameter1", 1),
      _wolfParam2("wolfParameter2", 1),
      _wolfParam3("wolfParameter3", 1),
      _prefactor("prefactor", 1)

{
    _coulombRadiusCutOff.h_view() = coulombRadiusCutOff;
    _kappa.h_view()               = kappa;
    _wolfParam1.h_view()          = wolfParameter1;
    _wolfParam2.h_view()          = wolfParameter2;
    _wolfParam3.h_view()          = wolfParameter3;
    _prefactor.h_view()           = prefactor;

    deep_copy(_coulombRadiusCutOff.d_view, _coulombRadiusCutOff.h_view);
    deep_copy(_kappa.d_view, _kappa.h_view);
    deep_copy(_wolfParam1.d_view, _wolfParam1.h_view);
    deep_copy(_wolfParam2.d_view, _wolfParam2.h_view);
    deep_copy(_wolfParam3.d_view, _wolfParam3.h_view);
    deep_copy(_prefactor.d_view, _prefactor.h_view);
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
KOKKOS_INLINE_FUNCTION
double KokkosCoulombWolf::calculate(
    const double distance,
    const double charge_i,
    const double charge_j,
    double      &force
) const
{
    const auto prefactor      = _prefactor.d_view();
    const auto kappa          = _kappa.d_view();
    const auto wolfParameter1 = _wolfParam1.d_view();
    const auto wolfParameter2 = _wolfParam2.d_view();
    const auto wolfParameter3 = _wolfParam3.d_view();
    const auto rcCutOff       = _coulombRadiusCutOff.d_view();

    const auto coulombPrefactor = charge_i * charge_j * prefactor;

    const auto kappaDistance        = kappa * distance;
    const auto kappaDistanceSquared = kappaDistance * kappaDistance;

    const auto erfcFactor = Kokkos::erfc(kappaDistance);
    const auto expFactor  = Kokkos::exp(-kappaDistanceSquared);

    auto energy  = erfcFactor / distance - wolfParameter1;
    energy      += wolfParameter3 * (distance - rcCutOff);

    auto scalarForce  = erfcFactor / (distance * distance);
    scalarForce      -= wolfParameter3;
    scalarForce      += wolfParameter2 * expFactor / distance;

    scalarForce *= coulombPrefactor;

    force += scalarForce;

    energy *= coulombPrefactor;
    return energy;
}

/**
 * @brief get the Coulomb radius cut off
 *
 * @return Kokkos::View<double>
 */
View<double> KokkosCoulombWolf::getCoulombRadiusCutOff() const
{
    return _coulombRadiusCutOff.d_view;
}