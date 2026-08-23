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

#include <unordered_set>

#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace configurator
{

    class HybridConfigurator
    {
       private:
        linearAlgebra::Vec3D _innerRegionCenter = {0.0};
        static inline bool   _molChangedZone    = false;

       public:
        void calculateInnerRegionCenter(molsys::SimulationBox &);
        void shiftAtomsToInnerRegionCenter(molsys::SimulationBox &);
        void shiftAtomsBackToInitialPositions(molsys::SimulationBox &);
        void assignHybridZones(molsys::SimulationBox &);
        void activateMolecules(molsys::SimulationBox &);
        void deactivateOuterMolecules(molsys::SimulationBox &);
        void activateSmoothingMolecules(molsys::SimulationBox &);
        void deactivateSmoothingMolecules(
            std::unordered_set<size_t> inactiveMolecules,
            molsys::SimulationBox &
        );
        void toggleMoleculeActivation(molsys::SimulationBox &);
        void calculateSmoothingFactors(molsys::SimulationBox &);

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] linearAlgebra::Vec3D getInnerRegionCenter() const;
        [[nodiscard]] static bool          getMoleculeChangedZone();

        static void setMoleculeChangedZone(bool);
    };

}   // namespace configurator

#endif   // _HYBRID_CONFIGURATOR_HPP_
