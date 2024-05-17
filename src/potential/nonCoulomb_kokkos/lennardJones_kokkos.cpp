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
    matrix_shared_pair &pairMatrix
)
{
    for (size_t i = 0; i < pairMatrix.rows(); ++i)
    {
        for (size_t j = 0; j < pairMatrix.cols(); ++j)
        {
            _radialCutoffs.h_view(i, j) = pairMatrix[i][j]->getRadialCutOff();
            _energyCutoffs.h_view(i, j) = pairMatrix[i][j]->getEnergyCutOff();
            _forceCutoffs.h_view(i, j)  = pairMatrix[i][j]->getForceCutOff();

            _c6.h_view(i, j) =
                dynamic_cast<LennardJonesPair *>(pairMatrix[i][j].get())
                    ->getC6();
            _c12.h_view(i, j) =
                dynamic_cast<LennardJonesPair *>(pairMatrix[i][j].get())
                    ->getC6();
        }
    }

    Kokkos::deep_copy(_radialCutoffs.d_view, _radialCutoffs.h_view);
    Kokkos::deep_copy(_energyCutoffs.d_view, _energyCutoffs.h_view);
    Kokkos::deep_copy(_forceCutoffs.d_view, _forceCutoffs.h_view);
    Kokkos::deep_copy(_c6.d_view, _c6.h_view);
    Kokkos::deep_copy(_c12.d_view, _c12.h_view);
}