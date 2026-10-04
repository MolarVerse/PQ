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

#ifndef _RESET_KINETICS_HPP_

#define _RESET_KINETICS_HPP_

#include <cstddef>
#include <mstd/types.hpp>

#include "resetKineticsSettings.hpp"
#include "vector3d.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace resetKinetics
{
    /**
     * @class ResetKinetics
     *
     * @brief base class for the reset of the kinetics - represents also class
     * for no reset
     *
     */
    class ResetKinetics
    {
       protected:
        mstd::ConstRef<ResetKineticsSettings> _settings;

       public:
        explicit ResetKinetics(const ResetKineticsSettings &settings);

        void reset(
            size_t step,
            physicalData::PhysicalData &,
            molsys::SimulationBox &
        ) const;
        void resetForces(size_t step, molsys::SimulationBox &) const;

        static void resetTemperature(
            molsys::SimulationBox &,
            double temperature
        );
        static void resetMomentum(
            molsys::SimulationBox &,
            const linalg::Vec3D &momentum
        );
        static void resetAngularMomentum(
            molsys::SimulationBox &,
            const linalg::Vec3D &angularMomentum
        );
    };

}   // namespace resetKinetics

#endif   // _RESET_KINETICS_HPP_
