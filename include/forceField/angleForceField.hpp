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

#ifndef _ANGLE_FORCE_FIELD_HPP_

#define _ANGLE_FORCE_FIELD_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "angle.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class Molecule;        // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace pot
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace pot

namespace forceField
{
    /**
     * @class BondForceField inherits from Bond
     *
     * @brief force field object for single angle
     *
     */
    class AngleForceField : public connectivity::Angle
    {
       private:
        AngleId _type;
        bool    _isLinker = false;

        double _equilibriumAngle = 0.0;
        double _forceConstant    = 0.0;

       public:
        AngleForceField(
            const std::vector<molsys::Molecule *> &molecules,
            const std::vector<size_t>             &atomIndices,
            const AngleId                          type
        );

        void calculateEnergyAndForces(
            const molsys::SimulationBox &simBox,
            physicalData::PhysicalData  &data,
            const pot::CoulombPotential &coulombPot,
            pot::NonCoulombPotential    &nonCoulombPot
        );

        /***************************
         * standard setter methods *
         ***************************/

        void setIsLinker(const bool isLinker);
        void setEquilibriumAngle(const double equilibriumAngle);
        void setForceConstant(const double forceConstant);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] AngleId getType() const;
        [[nodiscard]] bool    isLinker() const;
        [[nodiscard]] double  getEquilibriumAngle() const;
        [[nodiscard]] double  getForceConstant() const;
    };

}   // namespace forceField

#endif   // _ANGLE_FORCE_FIELD_HPP_
