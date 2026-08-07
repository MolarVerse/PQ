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

#ifndef _DISTANCE_CONSTRAINT_HPP_

#define _DISTANCE_CONSTRAINT_HPP_

#include <cstddef>

#include "bond.hpp"
#include "typeAliases.hpp"
#include "vector3d.hpp"

namespace constraints
{

    /**
     * @class DistanceConstraint inherits from Bond
     *
     * @brief constraint object for single bond length
     *
     * @details it performs the shake and rattle algorithm on a bond constraint
     *
     */
    class DistanceConstraint : public connectivity::Bond
    {
       private:
        double _lowerDistance;
        double _upperDistance;
        double _springConstant;
        double _dSpringConstantDt;

        double    _lowerEnergy = 0.0;
        double    _upperEnergy = 0.0;
        pq::Vec3D _force;

       public:
        DistanceConstraint(
            pq::Molecule *molecule1,
            pq::Molecule *molecule2,
            const size_t  atomIndex1,
            const size_t  atomIndex2,
            const double  lowerDistance,
            const double  upperDistance,
            const double  springConstant,
            const double  dSpringConstantDt
        );

        void applyDistanceConstraint(const pq::SimBox &, const double);

        [[nodiscard]] double getLowerDistance() const;
        [[nodiscard]] double getUpperDistance() const;
        [[nodiscard]] double getSpringConstant() const;
        [[nodiscard]] double getDSpringConstantDt() const;
        [[nodiscard]] double getLowerEnergy() const;
        [[nodiscard]] double getUpperEnergy() const;
    };

}   // namespace constraints

#endif   // _DISTANCE_CONSTRAINT_HPP_