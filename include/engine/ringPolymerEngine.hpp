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

#ifndef _RING_POLYMER_ENGINE_HPP_

#define _RING_POLYMER_ENGINE_HPP_

#include "engine.hpp"          // for Engine
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

#include <vector>   // for vector

namespace engine
{
    /**
     * @class RingPolymerEngine
     *
     * @details Contains all the information needed to run a ring polymer simulation
     *
     */
    class RingPolymerEngine : virtual public Engine
    {
      protected:
        std::vector<simulationBox::SimulationBox> _ringPolymerBeads;
        std::vector<physicalData::PhysicalData>   _ringPolymerBeadsPhysicalData;
        std::vector<physicalData::PhysicalData>   _averageRingPolymerBeadsPhysicalData;

      public:
        void writeOutput() override;

        void resizeRingPolymerBeadPhysicalData(const size_t numberOfBeads);

        void coupleRingPolymerBeads();
        void combineBeads();

        /************************
         * standard add methods *
         ************************/

        void addRingPolymerBead(const simulationBox::SimulationBox &bead) { _ringPolymerBeads.push_back(bead); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::vector<simulationBox::SimulationBox> &getRingPolymerBeads() { return _ringPolymerBeads; }
    };
}   // namespace engine

#endif   // _RING_POLYMER_ENGINE_HPP_