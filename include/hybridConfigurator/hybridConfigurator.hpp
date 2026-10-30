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

#include <mstd/types.hpp>
#include <unordered_set>

#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

class HybridSettings;

namespace configurator
{

    class HybridConfigurator
    {
       private:
        mstd::ConstRef<HybridSettings> _hybridSettings;
        linalg::Vec3D                  _innerRegionCenter = {0.0};
        static inline bool             _molChangedZone    = false;

       public:
        explicit HybridConfigurator(const HybridSettings &hybridSettings);

        void        calculateInnerRegionCenter(molsys::SimulationBox &);
        void        shiftAtomsToInnerRegionCenter(molsys::SimulationBox &);
        void        shiftAtomsBackToInitialPositions(molsys::SimulationBox &);
        void        assignHybridZones(molsys::SimulationBox &) const;
        static void activateMolecules(molsys::SimulationBox &);
        static void deactivateOuterMolecules(molsys::SimulationBox &);
        static void activateSmoothingMolecules(molsys::SimulationBox &);
        static void deactivateSmoothingMolecules(
            const std::unordered_set<size_t> &inactiveMolecules,
            molsys::SimulationBox &
        );
        static void toggleMoleculeActivation(molsys::SimulationBox &);
        void        calculateSmoothingFactors(molsys::SimulationBox &) const;

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] linalg::Vec3D getInnerRegionCenter() const;
        [[nodiscard]] static bool   getMoleculeChangedZone();

        static void setMoleculeChangedZone(bool);
    };

}   // namespace configurator

#endif   // _HYBRID_CONFIGURATOR_HPP_
