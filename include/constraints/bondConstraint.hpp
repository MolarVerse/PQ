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

#ifndef _BOND_CONSTRAINT_HPP_

#define _BOND_CONSTRAINT_HPP_

#include <cstddef>

#include "bond.hpp"
#include "vector3d.hpp"

namespace molsys
{
    class Molecule;        // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace constraints
{
    /**
     * @class BondConstraint inherits from Bond
     *
     * @brief constraint object for single bond length
     *
     * @details it performs the shake and rattle algorithm on a bond constraint
     *
     */
    class BondConstraint : public connectivity::Bond
    {
       private:
        double               _targetBondLength;
        linearAlgebra::Vec3D _shakeDistanceRef;

       public:
        BondConstraint(
            molsys::Molecule *molecule1,
            molsys::Molecule *molecule2,
            const size_t      atomIndex1,
            const size_t      atomIndex2,
            const double      bondLength
        );

        void calculateConstraintBondRef(const molsys::SimulationBox &);

        [[nodiscard]] double calculateDistanceDelta(
            const molsys::SimulationBox &
        ) const;
        [[nodiscard]] double calculateVelocityDelta() const;

        [[nodiscard]] bool applyShake(
            const molsys::SimulationBox &,
            const double
        );
        [[nodiscard]] bool applyRattle(const double);

        /***************************
         * standard setter methods *
         ***************************/

        void setShakeDistanceRef(const linearAlgebra::Vec3D &shakeDistanceRef);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double               getTargetBondLength() const;
        [[nodiscard]] linearAlgebra::Vec3D getShakeDistanceRef() const;
    };

}   // namespace constraints

#endif   // _BOND_CONSTRAINT_HPP_
