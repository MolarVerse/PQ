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

#ifndef _DIHEDRAL_FORCE_FIELD_HPP_

#define _DIHEDRAL_FORCE_FIELD_HPP_

#include <cstddef>
#include <vector>

#include "dihedral.hpp"
#include "typeAliases.hpp"

namespace forceField
{
    /**
     * @class DihedralForceField
     *
     * @brief Represents a dihedral between four atoms.
     *
     */
    class DihedralForceField : public connectivity::Dihedral
    {
       private:
        size_t _type;
        bool   _isLinker = false;

        double _forceConstant = 0.0;
        double _periodicity   = 0.0;
        double _phaseShift    = 0.0;

       public:
        DihedralForceField(
            const std::vector<pq::Molecule *> &molecules,
            const std::vector<size_t>         &atomIndices,
            const size_t                       type
        );

        void calculateEnergyAndForces(
            const pq::SimBox     &simBox,
            pq::PhysicalData     &data,
            const bool            isImproperDihedral,
            const pq::CoulombPot &coulombPot,
            pq::NonCoulombPot    &nonCoulombPot
        );

        /***************************
         * standard setter methods *
         ***************************/

        void setIsLinker(const bool isLinker);
        void setForceConstant(const double forceConstant);
        void setPeriodicity(const double periodicity);
        void setPhaseShift(const double phaseShift);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] bool isLinker() const;

        [[nodiscard]] size_t getType() const;
        [[nodiscard]] double getForceConstant() const;
        [[nodiscard]] double getPeriodicity() const;
        [[nodiscard]] double getPhaseShift() const;
    };

}   // namespace forceField

#endif   // _DIHEDRAL_FORCE_FIELD_HPP_
