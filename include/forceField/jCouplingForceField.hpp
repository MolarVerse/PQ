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

#ifndef _J_COUPLING_FORCE_FIELD_HPP_

#define _J_COUPLING_FORCE_FIELD_HPP_

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
    class JCouplingForceField : public connectivity::Dihedral
    {
       private:
        size_t _type;
        bool   _upperSymmetry = true;
        bool   _lowerSymmetry = true;

        double _J0;
        double _forceConstant;
        double _a;
        double _b;
        double _c;
        double _phaseShift;

       public:
        JCouplingForceField(
            const std::vector<pq::Molecule *> &molecules,
            const std::vector<size_t>         &atomIndices,
            const size_t                       type
        );

        void calculateEnergyAndForces(const pq::SimBox &, pq::PhysicalData &){
        };   // TODO: implement

        /***************************
         * standard setter methods *
         ***************************/

        void setUpperSymmetry(const bool boolean);
        void setLowerSymmetry(const bool boolean);

        void setJ0(const double J0);
        void setForceConstant(const double k);
        void setA(const double a);
        void setB(const double b);
        void setC(const double c);
        void setPhaseShift(const double phi);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getType() const;

        [[nodiscard]] bool getUpperSymmetry() const;
        [[nodiscard]] bool getLowerSymmetry() const;

        [[nodiscard]] double getJ0() const;
        [[nodiscard]] double getForceConstant() const;
        [[nodiscard]] double getA() const;
        [[nodiscard]] double getB() const;
        [[nodiscard]] double getC() const;
        [[nodiscard]] double getPhaseShift() const;
    };

}   // namespace forceField

#endif   // _J_COUPLING_FORCE_FIELD_HPP_
