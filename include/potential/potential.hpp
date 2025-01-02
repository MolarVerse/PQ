/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version.

    This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU General Public License for more
details.

    You should have received a copy of the GNU General
Public License along with this program.  If not, see
<http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _POTENTIAL_HPP_
#define _POTENTIAL_HPP_

/**
 * @file potential.hpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the main class definition for the potential
 * calculation. The class is used to calculate the forces and energies of the
 * inter non-bonded interactions for coulomb and non-coulomb potentials.
 *
 * @date 2024-12-09
 *
 */

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr, __shared_ptr_access, make_shared
#include <utility>   // for pair

#include "timer.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @brief A type definition for the function pointer to calculate the forces
     * and energies of the inter non-bonded interactions. This function pointer
     * is used to point to the actual template specialization of the function,
     * which is determined during runtime in the pre-simulation phase.
     *
     */
    using calcForcesPtr = void (*)(
        const Real *const   pos,
        Real *const         force,
        Real *const         shiftForce,
        const Real *const   charge,
        const Real *const   coulParams,
        const Real *const   nonCoulParams,
        const Real *const   ncCutOffs,
        const Real *const   boxParams,
        const size_t *const moleculeIndex,
        const size_t *const molTypes,
        const size_t *const atomTypes,
        Real               &totalCoulombEnergy,
        Real               &totalNonCoulombEnergy,
        const Real          coulCutOff,
        const size_t        nAtoms,
        const size_t        nAtomTypes,
        const size_t        nonCoulParamsOffset,
        const size_t        maxNumAtomTypes,
        const size_t        numMolTypes
    );

    /**
     * @class Potential
     *
     * @brief This class is used to calculate the forces and energies of the
     * inter non-bonded interactions for coulomb and non-coulomb potentials.
     *
     */
    class Potential : public timings::Timer
    {
       protected:
        pq::SharedCoulombPot    _coulombPotential;
        pq::SharedNonCoulombPot _nonCoulombPot;

        std::vector<Real> _nonCoulParams;
        std::vector<Real> _nonCoulCutOffs;
        std::vector<Real> _coulParams;

        size_t _nonCoulParamsOffset;
        size_t _nonCoulNumberOfTypes;

        size_t _maxNumAtomTypes;
        size_t _numMolTypes;

#ifdef __PQ_GPU__
        Real *_nonCoulParamsDevice;
        Real *_nonCoulCutOffsDevice;
        Real *_coulParamsDevice;
#endif

       public:
        virtual ~Potential() = default;

        void calculateForces(pq::SimBox &, pq::PhysicalData &, pq::CellList &);

        calcForcesPtr _cellListPtr;
        calcForcesPtr _bruteForcePtr;

        void setFunctionPointers(const bool isBoxOrthogonal);

        template <typename T>
        void makeCoulombPotential(T p);

        template <typename T>
        void makeNonCoulombPotential(T p);

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(const pq::SharedNonCoulombPot);
        void setCoulombParamVectors(std::vector<Real> coulParams);
        void setNonCoulombParamVectors(
            std::vector<Real> nonCoulParams,
            std::vector<Real> nonCoulCutOffs,
            const size_t      nonCoulParamsOffset,
            const size_t      nonCoulNumberOfTypes
        );

        void setMaxNumAtomTypes(const size_t maxNumAtomTypes);
        void setNumMolTypes(const size_t numMolTypes);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::CoulombPot         &getCoulombPotential() const;
        [[nodiscard]] pq::NonCoulombPot      &getNonCoulombPotential() const;
        [[nodiscard]] pq::SharedCoulombPot    getCoulombPotSharedPtr() const;
        [[nodiscard]] pq::SharedNonCoulombPot getNonCoulombPotSharedPtr() const;

        [[nodiscard]] Real *getNonCoulParamsPtr();
        [[nodiscard]] Real *getNonCoulCutOffsPtr();
        [[nodiscard]] Real *getCoulParamsPtr();
    };

    template <typename CoulombType, typename NonCoulombType, typename BoxType>
    void cellList(
        const Real *const   pos,
        Real *const         force,
        Real *const         shiftForce,
        const Real *const   charge,
        const Real *const   coulParams,
        const Real *const   nonCoulParams,
        const Real *const   ncCutOffs,
        const Real *const   boxParams,
        const size_t *const moleculeIndex,
        const size_t *const molTypes,
        const size_t *const atomTypes,
        Real               &totalCoulombEnergy,
        Real               &totalNonCoulombEnergy,
        const Real          coulCutOff,
        const size_t        nAtoms,
        const size_t        nAtomTypes,
        const size_t        nonCoulParamsOffset,
        const size_t        maxNumAtomTypes,
        const size_t        numMolTypes
    );

    template <typename CoulombType, typename NonCoulombType, typename BoxType>
    void bruteForce(
        const Real *const   pos,
        Real *const         force,
        Real *const         shiftForce,
        const Real *const   charge,
        const Real *const   coulParams,
        const Real *const   nonCoulParams,
        const Real *const   ncCutOffs,
        const Real *const   boxParams,
        const size_t *const moleculeIndex,
        const size_t *const molTypes,
        const size_t *const atomTypes,
        Real               &totalCoulombEnergy,
        Real               &totalNonCoulombEnergy,
        const Real          coulCutOff,
        const size_t        nAtoms,
        const size_t        nAtomTypes,
        const size_t        nonCoulParamsOffset,
        const size_t        maxNumAtomTypes,
        const size_t        numMolTypes
    );

    template <typename BoxType>
    void inline image(
        const auto &boxParams,
        auto       &dx,
        auto       &dy,
        auto       &dz,
        auto       &tx,
        auto       &ty,
        auto       &tz
    );

    template <typename CoulombType>
    void inline calculateCoulomb(
        auto             &coulombEnergy,
        auto             &localForce,
        const auto        distance,
        const auto        coulombPreFactor,
        const auto        coulCutOff,
        const auto *const coulParams
    );

    template <typename NonCoulombType>
    void calculateNonCoulombEnergy(
        auto             &nonCoulombEnergy,
        auto             &localForce,
        const auto        distance,
        const auto        distanceSquared,
        const auto        rncCutOff,
        const auto *const nonCoulParams
    );

}   // namespace potential

// clang-format off
#include "potentialHandleTypes.inl"      // DO NOT MOVE THIS LINE
#include "potential.inl"             // DO NOT MOVE THIS LINE
#include "potentialBruteForce.inl"   // DO NOT MOVE THIS LINE
// clang-format on

#endif   // _POTENTIAL_HPP      _