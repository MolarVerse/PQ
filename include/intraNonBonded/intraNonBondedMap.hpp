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

#ifndef _INTRA_NON_BONDED_MAP_HPP_

#define _INTRA_NON_BONDED_MAP_HPP_

#include <cstddef>   // for size_t
#include <utility>   // for pair
#include <vector>    // for vector

#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "typeAliases.hpp"
#include "vector3d.hpp"   // for Vec3D

namespace intraNonBonded
{
    /**
     * @class IntraNonBondedMap
     *
     * @brief defines a map for a single molecule to its intra non bonded
     * interactions
     */
    class IntraNonBondedMap
    {
       private:
        pq::Molecule            *_molecule;
        IntraNonBondedContainer *_intraNonBondedContainer;

       public:
        explicit IntraNonBondedMap(
            pq::Molecule            *molecule,
            IntraNonBondedContainer *intraNonBondedType
        );

        void calculate(
            const pq::CoulombPot *coulPot,
            pq::NonCoulombPot    *nonCoulPot,
            const pq::SimBox     &simBox,
            pq::PhysicalData     &data
        ) const;

        [[nodiscard]] std::pair<double, double> calculateSingleInteraction(
            const size_t          atomIdx1,
            const int             atomIdx2,
            const pq::Vec3D      &box,
            pq::PhysicalData     &data,
            const pq::CoulombPot *coulPot,
            pq::NonCoulombPot    *nonCoulPot
        ) const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] IntraNonBondedContainer *getIntraNonBondedType() const;
        [[nodiscard]] pq::Molecule            *getMolecule() const;
        [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const;
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_MAP_HPP_