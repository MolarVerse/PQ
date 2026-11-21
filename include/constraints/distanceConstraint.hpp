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

#ifndef CONSTRAINTS__DISTANCE_CONSTRAINT_HPP
#define CONSTRAINTS__DISTANCE_CONSTRAINT_HPP

#include "bond.hpp"
#include "strongTypes.hpp"
#include "unit_types.hpp"

namespace molsys
{
    class Molecule;        // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace constraints
{

    /**
     * @brief DistanceConstraint inherits from Bond
     *
     * @details it performs the shake and rattle algorithm on a bond constraint
     *
     */
    class DistanceConstraint : public connectivity::Bond
    {
       private:
        Distance             _lowerDistance;
        Distance             _upperDistance;
        ForceConstant        _springConstant;
        ForceConstantPerTime _dSpringConstantDt;

        EnergyPerMol _lowerEnergy = EnergyPerMol::zero();
        EnergyPerMol _upperEnergy = EnergyPerMol::zero();

       public:
        DistanceConstraint(
            molsys::Molecule           *molecule1,
            molsys::Molecule           *molecule2,
            AtomIndex                   atomIndex1,
            AtomIndex                   atomIndex2,
            const Distance             &lowerDistance,
            const Distance             &upperDistance,
            const ForceConstant        &springConstant,
            const ForceConstantPerTime &dSpringConstantDt
        );

        void applyDistanceConstraint(
            const molsys::SimulationBox &,
            const Time &
        );

        CONST_GETTER(_lowerDistance);
        CONST_GETTER(_upperDistance);
        CONST_GETTER(_springConstant);
        CONST_GETTER(_dSpringConstantDt);
        CONST_GETTER(_lowerEnergy);
        CONST_GETTER(_upperEnergy);
    };

}   // namespace constraints

#endif   // CONSTRAINTS__DISTANCE_CONSTRAINT_HPP
