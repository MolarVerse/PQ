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

#ifndef __POTENTIAL_HANDLE_TYPES_INL__
#define __POTENTIAL_HANDLE_TYPES_INL__

/**
 * @file potentialHandleTypes.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief  This file is used to define the template functions for the inter
 * non-bonded routines. The functions are just wrappers for the actual functions
 * in order to reduce the verbosity of the actual non-bonded routines.
 *
 * @date 2024-12-09
 */

#include <cassert>
#include <type_traits>

#include "buckingham.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "lennardJones.hpp"
#include "morse.hpp"
#include "orthorhombicBox.hpp"
#include "potential.hpp"
#include "triclinicBox.hpp"

namespace potential
{
    /**
     * @brief image
     *
     * @details This function is used to apply periodic boundary conditions to
     * the coordinates of the atoms. This is just a template function
     * functioning as a wrapper for the actual image function.
     *
     * @tparam BoxType the type of the box to use
     * @param boxParams
     * @param dx
     * @param dy
     * @param dz
     * @param tx
     * @param ty
     * @param tz
     */
    template <typename BoxType>
    void inline image(
        const Real* __restrict__ const boxParams,
        Real& dx,
        Real& dy,
        Real& dz,
        Real& tx,
        Real& ty,
        Real& tz
    )
    {
        if constexpr (std::is_same_v<BoxType, simulationBox::OrthorhombicBox>)
            simulationBox::imageOrthoRhombic(boxParams, dx, dy, dz, tx, ty, tz);
        else if constexpr (std::is_same_v<BoxType, simulationBox::TriclinicBox>)
            simulationBox::imageTriclinic(boxParams, dx, dy, dz, tx, ty, tz);
        else
            static_assert(
                std::is_same_v<BoxType, void>,
                "Unsupported Box type"
            );
    }

    /**
     * @brief calculateCoulomb
     *
     * @details This function is used to calculate the Coulomb potential
     * between two atoms. This is just a template function functioning as a
     * wrapper for the actual calculateCoulombPotential function.
     *
     * @tparam CoulombType the type of the Coulomb potential to use
     * @param coulombEnergy
     * @param localForce
     * @param distance
     * @param coulombPreFactor
     * @param coulCutOff
     * @param coulParams
     */
    template <typename CoulombType>
    void inline calculateCoulombPotential(
        Real&      coulombEnergy,
        Real&      localForce,
        const Real distance,
        const Real coulombPreFactor,
        const Real coulCutOff,
        const Real* __restrict__ const coulParams
    )
    {
        if constexpr (std::is_same_v<CoulombType, CoulombShiftedPotential>)
            coulombEnergy = calculateCoulombShiftedPotential(
                localForce,
                distance,
                coulombPreFactor,
                coulCutOff,
                coulParams
            );
        else if constexpr (std::is_same_v<CoulombType, CoulombWolf>)
            coulombEnergy = calculateCoulombWolfPotential(
                localForce,
                distance,
                coulombPreFactor,
                coulCutOff,
                coulParams
            );
        else
            static_assert(
                std::is_same_v<CoulombType, void>,
                "Unsupported Coulomb type"
            );
    }

    /**
     * @brief calculateNonCoulombEnergy
     *
     * @details This function is used to calculate the non-Coulomb potential
     * between two atoms. This is just a template function functioning as a
     * wrapper for the actual calculateNonCoulombEnergy function.
     *
     * @tparam NonCoulombType the type of the non-Coulomb potential to use
     * @param nonCoulombEnergy
     * @param localForce
     * @param distance
     * @param distanceSquared
     * @param rncCutOff
     * @param nonCoulParams
     */
    template <typename NonCoulombType>
    void calculateNonCoulombEnergy(
        Real&                       nonCoulombEnergy,
        Real&                       localForce,
        const Real                  distance,
        [[maybe_unused]] const Real distanceSquared,   // not needed for Morse
        const Real                  rncCutOff,
        const Real* __restrict__ const nonCoulParams
    )
    {
        if constexpr (std::is_same_v<NonCoulombType, LennardJonesFF>)
            nonCoulombEnergy = calculateLennardJones(
                localForce,
                distance,
                distanceSquared,
                rncCutOff,
                nonCoulParams
            );
        else if constexpr (std::is_same_v<NonCoulombType, BuckinghamFF>)
            nonCoulombEnergy = calculateBuckingham(
                localForce,
                distance,
                distanceSquared,
                rncCutOff,
                nonCoulParams
            );
        else if constexpr (std::is_same_v<NonCoulombType, MorseFF>)
            nonCoulombEnergy =
                calculateMorse(localForce, distance, rncCutOff, nonCoulParams);
        else
            static_assert(
                std::is_same_v<NonCoulombType, void>,
                "Unsupported NonCoulomb type"
            );
    }

    template <typename NonCoulombType>
    static void inline fetchCombinedIndices(
        const size_t* __restrict__ const atomTypes,
        const size_t* __restrict__ const molTypes,
        const size_t mol_i,
        const size_t mol_j,
        const size_t atom_i,
        const size_t atom_j,
        const size_t nAtomTypes,
        const size_t nMolTypes,
        const size_t nonCoulParamsOffset,
        int&         comCutoffIndex,
        int&         comIndex
    )
    {
        if constexpr (std::is_same_v<NonCoulombType, LennardJonesFF> ||
                      std::is_same_v<NonCoulombType, BuckinghamFF> ||
                      std::is_same_v<NonCoulombType, MorseFF>)
        {
            comCutoffIndex = atomTypes[atom_i] * nAtomTypes + atomTypes[atom_j];
            comIndex       = comCutoffIndex * nonCoulParamsOffset;
        }
        else if constexpr (std::is_same_v<NonCoulombType, LennardJonesGuff> ||
                           std::is_same_v<NonCoulombType, BuckinghamGuff> ||
                           std::is_same_v<NonCoulombType, MorseGuff>)
        {
            auto index      = molTypes[mol_i] * nMolTypes;
            index          += molTypes[mol_j] * nMolTypes;
            index          += atomTypes[atom_i] * nMolTypes;
            index          += atomTypes[atom_j];
            comCutoffIndex  = index;
            comIndex        = index * nonCoulParamsOffset;
        }
        else
            static_assert(
                std::is_same_v<NonCoulombType, void>,
                "The nonCoulomb potential is not implemented yet"
            );
    }
}   // namespace potential

#endif   // __POTENTIAL_HANDLE_TYPES_INL__