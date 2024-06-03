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

        KOKKOS_FUNCTION double getRadialCutoff(const size_t i, const size_t j)
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

        KOKKOS_INLINE_FUNCTION double calculate(
            const double distance,
            double      &force,
            const size_t vdWType_i,
            const size_t vdWType_j
        ) const
        {
            // calculate r^12 and r^6
            const auto distanceSquared = distance * distance;
            const auto distanceSixth =
                distanceSquared * distanceSquared * distanceSquared;
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
    };
}   // namespace potential

#endif   // _KOKKOS_LENNARD_JONES_PAIR_HPP_