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

#ifndef _POTENTIAL_CELL_LIST_HPP_

#define _POTENTIAL_CELL_LIST_HPP_

#include "potential.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class CellList;   // forward declaration
}   // namespace molsys

namespace potential
{
    /**
     * @class PotentialCellList
     *
     * @brief cell list implementation of the potential
     *
     */
    class PotentialCellList : public Potential
    {
       public:
        ~PotentialCellList() override;

        void calculateForces(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::CellList &
        ) override;

        void calculateCoreToOuterForces(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::CellList &
        ) override;

        void calculateLayerToOuterForces(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::CellList &
        ) override;

        void calculateOuterToOuterForces(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::CellList &
        ) override;

        void calculateHotspotSmoothingMMForces(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::CellList &
        ) override;

        std::shared_ptr<Potential> clone() const override;
    };

}   // namespace potential

#endif   // _POTENTIAL_CELL_LIST_HPP_
