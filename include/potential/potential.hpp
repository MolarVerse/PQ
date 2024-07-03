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

       public:
        virtual ~Potential() = default;

        virtual void calculateForces(pq::SimBox &, pq::PhysicalData &, pq::CellList &) = 0;
        virtual pq::SharedPotential clone() const = 0;

        std::pair<double, double> calculateSingleInteraction(
            const pq::Box &,
            pq::Molecule &,
            pq::Molecule &,
            const size_t,
            const size_t
        ) const;

        template <typename T>
        void makeCoulombPotential(T p);

        template <typename T>
        void makeNonCoulombPotential(T nonCoulombPot);

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(const pq::SharedNonCoulombPot);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::CoulombPot         &getCoulombPotential() const;
        [[nodiscard]] pq::NonCoulombPot      &getNonCoulombPotential() const;
        [[nodiscard]] pq::SharedCoulombPot    getCoulombPotSharedPtr() const;
        [[nodiscard]] pq::SharedNonCoulombPot getNonCoulombPotSharedPtr() const;
    };

}   // namespace potential

#include "potential.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _POTENTIAL_HPP_