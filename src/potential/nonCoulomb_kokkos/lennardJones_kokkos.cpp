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

#include "lennardJones_kokkos.hpp"

using namespace potential;
using namespace Kokkos;
using namespace linearAlgebra;

/**
 * @brief constructor
 */
KokkosLennardJones::KokkosLennardJones(size_t numAtomTypes)
    : _radialCutoffs("radialCutoffs", numAtomTypes, numAtomTypes),
      _energyCutoffs("energyCutoffs", numAtomTypes, numAtomTypes),
      _forceCutoffs("forceCutoffs", numAtomTypes, numAtomTypes),
      _c6("c6", numAtomTypes, numAtomTypes),
      _c12("c12", numAtomTypes, numAtomTypes)
{
}

/**
 * @brief transfer from non Coulomb pair matrix
 *
 * @param pairMatrix non Coulomb pair matrix
 */
void KokkosLennardJones::transferFromNonCoulombPairMatrix(
    Matrix<std::shared_ptr<NonCoulombPair>> &pairMatrix
)
{
    for (size_t i = 0; i < pairMatrix.rows(); ++i)
        for (size_t j = 0; j < pairMatrix.cols(); ++j)
        {
            _radialCutoffs.h_view(i, j) = pairMatrix(i, j)->getRadialCutOff();
            _energyCutoffs.h_view(i, j) = pairMatrix(i, j)->getEnergyCutOff();
            _forceCutoffs.h_view(i, j)  = pairMatrix(i, j)->getForceCutOff();

            auto *nonCoulPair = pairMatrix(i, j).get();
            auto *pair        = dynamic_cast<LennardJonesPair *>(nonCoulPair);

            _c6.h_view(i, j)  = pair->getC6();
            _c12.h_view(i, j) = pair->getC12();
        }

    deep_copy(_radialCutoffs.d_view, _radialCutoffs.h_view);
    deep_copy(_energyCutoffs.d_view, _energyCutoffs.h_view);
    deep_copy(_forceCutoffs.d_view, _forceCutoffs.h_view);
    deep_copy(_c6.d_view, _c6.h_view);
    deep_copy(_c12.d_view, _c12.h_view);
}

/**
 * @brief Calculate the Lennard-Jones (12-6) energy and forces
 * between two atoms and add the forces to the force vector.
 *
 * @param distance distance between atoms
 * @param force force to store
 * @param vdWType_i van der Waals type of atom i
 * @param vdWType_j van der Waals type of atom j
 * @return double energy
 */
KOKKOS_INLINE_FUNCTION double KokkosLennardJones::calculate(
    const double distance,
    double      &force,
    const size_t vdWType_i,
    const size_t vdWType_j
) const
{
    // calculate r^12 and r^6
    const auto distanceThird   = distance * distance * distance;
    const auto distanceSixth   = distanceThird * distanceThird;
    const auto distanceTwelfth = distanceSixth * distanceSixth;

    const auto c12     = _c12.d_view(vdWType_i, vdWType_j);
    const auto c6      = _c6.d_view(vdWType_i, vdWType_j);
    const auto eCutoff = _energyCutoffs.d_view(vdWType_i, vdWType_j);
    const auto fCutoff = _forceCutoffs.d_view(vdWType_i, vdWType_j);
    const auto rCutoff = _radialCutoffs.d_view(vdWType_i, vdWType_j);

    // calculate energy
    auto energy  = c12 / distanceTwelfth;
    energy      += c6 / distanceSixth;
    energy      -= eCutoff;
    energy      -= fCutoff * (rCutoff - distance);

    // calculate force
    auto scalarForce  = 12.0 * c12 / (distanceTwelfth * distance);
    scalarForce      += 6.0 * c6 / (distanceSixth * distance);
    scalarForce      -= fCutoff;

    force += scalarForce;

    return energy;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get dual view of radial cutoffs
 *
 * @return DualView<double **>&
 */
DualView<double **> &KokkosLennardJones::getRadialCutoffs()
{
    return _radialCutoffs;
}

/**
 * @brief get dual view of energy cutoffs
 *
 * @return DualView<double **>&
 */
DualView<double **> &KokkosLennardJones::getEnergyCutoffs()
{
    return _energyCutoffs;
}

/**
 * @brief get dual view of force cutoffs
 *
 * @return DualView<double **>&
 */
DualView<double **> &KokkosLennardJones::getForceCutoffs()
{
    return _forceCutoffs;
}

/**
 * @brief get dual view of C6
 *
 * @return DualView<double **>&
 */
DualView<double **> &KokkosLennardJones::getC6() { return _c6; }

/**
 * @brief get dual view of C12
 *
 * @return DualView<double **>&
 */
DualView<double **> &KokkosLennardJones::getC12() { return _c12; }

/**
 * @brief get radial cutoff by index
 *
 * @param i
 * @param j
 * @return KOKKOS_FUNCTION
 */
KOKKOS_FUNCTION double KokkosLennardJones::getRadialCutoff(
    const size_t i,
    const size_t j
) const
{
    return _radialCutoffs.d_view(i, j);
}

/**
 * @brief get energy cutoff by index
 *
 * @param i
 * @param j
 * @return double
 */
double KokkosLennardJones::getEnergyCutoff(const size_t i, const size_t j) const
{
    return _energyCutoffs.d_view(i, j);
}

/**
 * @brief get force cutoff by index
 *
 * @param i
 * @param j
 * @return double
 */
double KokkosLennardJones::getForceCutoff(const size_t i, const size_t j) const
{
    return _forceCutoffs.d_view(i, j);
}

/**
 * @brief get C6 by index
 *
 * @param i
 * @param j
 * @return double
 */
double KokkosLennardJones::getC6(const size_t i, const size_t j) const
{
    return _c6.d_view(i, j);
}

/**
 * @brief get C12 by index
 *
 * @param i
 * @param j
 * @return double
 */
double KokkosLennardJones::getC12(const size_t i, const size_t j) const
{
    return _c12.d_view(i, j);
}