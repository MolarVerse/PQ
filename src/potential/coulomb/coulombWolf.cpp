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

#include "coulombWolf.hpp"

#include <cmath>   // for exp, sqrt, erfc

#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

using namespace potential;
using namespace constants;

/**
 * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
 *
 * @details this constructor calculates automatically the three need wolf
 * parameters from kappa in order to gain speed
 *
 * @param coulRC - coulomb radius cut off
 * @param kappa
 */
CoulombWolf::CoulombWolf(const double coulRC, const double kappa)
    : CoulombPotential(coulRC)
{
    _kappa      = kappa;
    _wolfParam1 = ::erfc(_kappa * coulRC) / coulRC;
    _wolfParam2 = 2.0 * _kappa / ::sqrt(M_PI);

    const auto kappaSquared  = _kappa * _kappa;
    const auto coulRCSquared = coulRC * coulRC;
    const auto expFactor     = ::exp(-kappaSquared * coulRCSquared);

    _wolfParam3  = _wolfParam1 / coulRC;
    _wolfParam3 += _wolfParam2 * expFactor / coulRC;
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
std::pair<double, double> CoulombWolf::calculate(
    const double distance,
    const double chargeProduct
) const
{
    const auto coulombPrefactor = chargeProduct * _COULOMB_PREFACTOR_;

    const auto kappaDistance = _kappa * distance;
    const auto erfcFactor    = ::erfc(kappaDistance);
    const auto expFactor     = ::exp(-kappaDistance * kappaDistance);

    auto energy  = erfcFactor / distance - _wolfParam1;
    energy      += _wolfParam3 * (distance - _coulombRadiusCutOff);

    auto force  = erfcFactor / (distance * distance);
    force      += _wolfParam2 * expFactor / distance;
    force      -= _wolfParam3;

    energy *= coulombPrefactor;
    force  *= coulombPrefactor;

    return {energy, force};
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the kappa parameter
 *
 * @param kappa
 */
void CoulombWolf::setKappa(const double kappa) { _kappa = kappa; }

/**
 * @brief set the wolf parameter 1
 *
 * @param wolfParam1
 */
void CoulombWolf::setWolfParameter1(const double wolfParam1)
{
    _wolfParam1 = wolfParam1;
}

/**
 * @brief set the wolf parameter 2
 *
 * @param wolfParam2
 */
void CoulombWolf::setWolfParameter2(const double wolfParam2)
{
    _wolfParam2 = wolfParam2;
}

/**
 * @brief set the wolf parameter 3
 *
 * @param wolfParam3
 */
void CoulombWolf::setWolfParameter3(const double wolfParam3)
{
    _wolfParam3 = wolfParam3;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the kappa parameter
 *
 * @return double
 */
[[nodiscard]] double CoulombWolf::getKappa() const { return _kappa; }

/**
 * @brief get the wolf parameter 1
 *
 * @return double
 */
[[nodiscard]] double CoulombWolf::getWolfParameter1() const
{
    return _wolfParam1;
}

/**
 * @brief get the wolf parameter 2
 *
 * @return double
 */
[[nodiscard]] double CoulombWolf::getWolfParameter2() const
{
    return _wolfParam2;
}

/**
 * @brief get the wolf parameter 3
 *
 * @return double
 */
[[nodiscard]] double CoulombWolf::getWolfParameter3() const
{
    return _wolfParam3;
}

/**
 * @brief copy the parameter vector
 *
 * @return std::vector<Real>
 */
std::vector<Real> CoulombWolf::copyParamsVector()
{
    return std::vector<Real>{
        _coulombEnergyCutOff,
        _coulombForceCutOff,
        _kappa,
        _wolfParam1,
        _wolfParam2,
        _wolfParam3
    };
}