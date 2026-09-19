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

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class Molecule;        // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

struct TestForceFieldUtils;   // forward declaration

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
        friend struct ::TestForceFieldUtils;

       private:
        size_t _type;
        bool   _upperSymmetry = true;
        bool   _lowerSymmetry = true;

        std::optional<JCouplingParams> _params;

       public:
        JCouplingForceField(
            const std::vector<molsys::Molecule *> &molecules,
            const std::vector<AtomIndex>          &atomIndices,
            size_t                                 type
        );

        void calculateEnergyAndForces(
            const molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/
        )
        {
        }   // TODO: implement

        /***************************
         * standard setter methods *
         ***************************/

        void setUpperSymmetry(bool boolean);
        void setLowerSymmetry(bool boolean);

        void setParams(const JCouplingParams &params);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getType() const;

        [[nodiscard]] bool getUpperSymmetry() const;
        [[nodiscard]] bool getLowerSymmetry() const;
    };

}   // namespace forceField

#endif   // _J_COUPLING_FORCE_FIELD_HPP_
