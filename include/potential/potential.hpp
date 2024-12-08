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

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr, __shared_ptr_access, make_shared
#include <utility>   // for pair

#include "timer.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class Potential
     *
     * @brief base class for all potential routines
     *
     * @details
     * possible options:
     * - brute force
     * - cell list
     *
     * @note _nonCoulPairsVec is just a container to store the
     * nonCoulombicPairs for later processing
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

#ifdef __PQ_GPU__
        Real *_nonCoulParamsDevice;
        Real *_nonCoulCutOffsDevice;
        Real *_coulParamsDevice;
#endif

       public:
        virtual ~Potential() = default;

        void calculateForces(pq::SimBox &, pq::PhysicalData &, pq::CellList &);

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
            const size_t        nonCoulParamsOffset
        );

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
        void setNonCoulombParamVectors(
            const std::vector<Real> nonCoulParams,
            const std::vector<Real> nonCoulCutOffs,
            const size_t            nonCoulParamsOffset,
            const size_t            nonCoulNumberOfTypes
        );
        void setCoulombParamVectors(const std::vector<Real> coulParams);

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
        const size_t        nonCoulParamsOffset
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
        const size_t        nonCoulParamsOffset
    );

}   // namespace potential

#include "potential.tpp.hpp"             // DO NOT MOVE THIS LINE
#include "potentialBruteForce.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _POTENTIAL_HPP_