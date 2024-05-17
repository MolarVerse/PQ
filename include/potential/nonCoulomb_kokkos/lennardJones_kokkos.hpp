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
        Kokkos::DualView<double **> _radialCutoffs;
        Kokkos::DualView<double **> _energyCutoffs;
        Kokkos::DualView<double **> _forceCutoffs;
        Kokkos::DualView<double **> _c6;
        Kokkos::DualView<double **> _c12;

       public:
        KokkosLennardJones(size_t numAtomTypes);

        KokkosLennardJones()  = default;
        ~KokkosLennardJones() = default;

        void transferFromNonCoulombPairMatrix(matrix_shared_pair &pairMatrix);
        [[nodiscard]] Kokkos::DualView<double **> &getRadialCutoffs()
        {
            return _radialCutoffs;
        }
        [[nodiscard]] Kokkos::DualView<double **> &getEnergyCutoffs()
        {
            return _energyCutoffs;
        }
        [[nodiscard]] Kokkos::DualView<double **> &getForceCutoffs()
        {
            return _forceCutoffs;
        }
        [[nodiscard]] Kokkos::DualView<double **> &getC6() { return _c6; }
        [[nodiscard]] Kokkos::DualView<double **> &getC12() { return _c12; }

        [[nodiscard]] double getRadialCutoff(const size_t i, const size_t j)
            const
        {
            return _radialCutoffs.d_view(i, j);
        }
        [[nodiscard]] double getEnergyCutoff(const size_t i, const size_t j)
            const
        {
            return _energyCutoffs.d_view(i, j);
        }
        [[nodiscard]] double getForceCutoff(const size_t i, const size_t j)
            const
        {
            return _forceCutoffs.d_view(i, j);
        }
        [[nodiscard]] double getC6(const size_t i, const size_t j) const
        {
            return _c6.d_view(i, j);
        }
        [[nodiscard]] double getC12(const size_t i, const size_t j) const
        {
            return _c12.d_view(i, j);
        }

        KOKKOS_INLINE_FUNCTION
        static double calculatePairEnergy(
            const double distance,
            const double dxyz[3],
            double      *force_i,
            const size_t vdWType_i,
            const size_t vdWType_j
        )
        {
            auto nonCoulombicEnergy = 0.0;

            return nonCoulombicEnergy;
        }
    };
}   // namespace potential

#endif   // _KOKKOS_LENNARD_JONES_PAIR_HPP_