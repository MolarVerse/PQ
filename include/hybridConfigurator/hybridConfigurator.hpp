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

#ifndef _HYBRID_CONFIGURATOR_HPP_

#define _HYBRID_CONFIGURATOR_HPP_

#include <unordered_set>   // for unordered_set

#include "simulationBox.hpp"
#include "typeAliases.hpp"

namespace configurator
{

    class HybridConfigurator
    {
       private:
        pq::Vec3D _innerRegionCenter        = {0.0, 0.0, 0.0};
        int       _numberSmoothingMolecules = 0;

       public:
        void calculateInnerRegionCenter(pq::SimBox &);
        void shiftAtomsToInnerRegionCenter(pq::SimBox &);
        void shiftAtomsBackToInitialPositions(pq::SimBox &);
        void assignHybridZones(pq::SimBox &);
        void deactivateMoleculesForInnerCalculation(
            std::unordered_set<size_t>,
            pq::SimBox &
        );
        void calculateSmoothingFactors(pq::SimBox &);

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] int getNumberSmoothingMolecules();

        void setNumberSmoothingMolecules(int);
    };

}   // namespace configurator

#endif   // _HYBRID_CONFIGURATOR_HPP_