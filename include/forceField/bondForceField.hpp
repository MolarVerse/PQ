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

#ifndef _BOND_FORCE_FIELD_HPP_

#define _BOND_FORCE_FIELD_HPP_

#include <cstddef>

#include "bond.hpp"
#include "typeAliases.hpp"

namespace forceField
{
    /**
     * @class BondForceField inherits from Bond
     *
     * @brief force field object for single bond length
     *
     */
    class BondForceField : public connectivity::Bond
    {
       private:
        size_t _type;
        bool   _isLinker = false;

        double _equilBondLength;
        double _forceConstant;

       public:
        BondForceField(
            pq::Molecule *molecule1,
            pq::Molecule *molecule2,
            const size_t  atomIndex1,
            const size_t  atomIndex2,
            const size_t  type
        );

        void calculateEnergyAndForces(
            const pq::SimBox     &simBox,
            pq::PhysicalData     &data,
            const pq::CoulombPot &coulombPot,
            pq::NonCoulombPot    &nonCoulombPot
        );

        /***************************
         * standard setter methods *
         ***************************/

        void setIsLinker(const bool isLinker);
        void setEquilibriumBondLength(const double equilibriumBondLength);
        void setForceConstant(const double forceConstant);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getType() const;
        [[nodiscard]] bool   isLinker() const;
        [[nodiscard]] double getEquilibriumBondLength() const;
        [[nodiscard]] double getForceConstant() const;
    };

}   // namespace forceField

#endif   // _BOND_FORCE_FIELD_HPP_