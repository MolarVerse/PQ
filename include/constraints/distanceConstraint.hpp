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
#include "vector3d.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
}   // namespace simulationBox

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

        double               _lowerEnergy = 0.0;
        double               _upperEnergy = 0.0;
        linearAlgebra::Vec3D _force       = {0.0};

       public:
        DistanceConstraint(
            simulationBox::Molecule *molecule1,
            simulationBox::Molecule *molecule2,
            const size_t             atomIndex1,
            const size_t             atomIndex2,
            const double             lowerDistance,
            const double             upperDistance,
            const double             springConstant,
            const double             dSpringConstantDt
        )
            : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2),
              _lowerDistance(lowerDistance),
              _upperDistance(upperDistance),
              _springConstant(springConstant),
              _dSpringConstantDt(dSpringConstantDt){};

        void applyDistanceConstraint(
            const simulationBox::SimulationBox &simulationBox,
            const double                        dt
        );

        [[nodiscard]] double getLowerDistance() const { return _lowerDistance; }
        [[nodiscard]] double getUpperDistance() const { return _upperDistance; }

        [[nodiscard]] double getSpringConstant() const
        {
            return _springConstant;
        }
        [[nodiscard]] double getDSpringConstantDt() const
        {
            return _dSpringConstantDt;
        }

        [[nodiscard]] double getLowerEnergy() const { return _lowerEnergy; }
        [[nodiscard]] double getUpperEnergy() const { return _upperEnergy; }
    };

}   // namespace constraints

#endif   // _DISTANCE_CONSTRAINT_HPP_