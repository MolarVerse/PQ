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

#ifndef _KOKKOS_LENNARD_JONES_PAIR_HPP_

#define _KOKKOS_LENNARD_JONES_PAIR_HPP_

#include <Kokkos_DualView.hpp>

#include "forceFieldNonCoulomb.hpp"   // for matrix_shared_pair
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "matrix.hpp"                 // for matrix

namespace potential
{
    /**
     * @class KokkosLennardJones
     *
     * @brief containing all information about the Lennard-Jones potential
     */
    class KokkosLennardJones
    {
       private:
        Kokkos::DualView<double**> _radialCutoffs;
        Kokkos::DualView<double**> _energyCutoffs;
        Kokkos::DualView<double**> _forceCutoffs;
        Kokkos::DualView<double**> _c6;
        Kokkos::DualView<double**> _c12;

       public:
        KokkosLennardJones(size_t numAtomTypes);

        KokkosLennardJones()  = default;
        ~KokkosLennardJones() = default;

        void transferFromNonCoulombPairMatrix(matrix_shared_pair& pairMatrix);
    };
}   // namespace potential

#endif   // _KOKKOS_LENNARD_JONES_PAIR_HPP_