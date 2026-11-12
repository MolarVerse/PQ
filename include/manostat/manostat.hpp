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

#ifndef MANOSTAT__MANOSTAT_HPP
#define MANOSTAT__MANOSTAT_HPP

#include <mstd/types.hpp>

#include "unit_types.hpp"

namespace molsys
{
    class Box;             // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

class ManostatSettings;   // forward declaration

namespace manostat
{
    /**
     * @class Manostat
     *
     * @brief Manostat is a base class for all manostats
     *
     */
    class Manostat
    {
       private:
        mstd::ConstRef<ManostatSettings> _settings;
        PressureTensor3D                 _pressureTensor;
        Pressure                         _pressure;

       public:
        explicit Manostat(const ManostatSettings &settings);
        virtual ~Manostat() = default;

        void calculatePressure(
            const molsys::SimulationBox &,
            physicalData::PhysicalData &
        );

        virtual void applyManostat(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const Temperature &targetTemperature
        );

        static void rotateMu(QuantityTensor3D &mu);

       protected:
        static void _validateScaling(
            const molsys::SimulationBox &,
            const QuantityTensor3D &
        );

        [[nodiscard]]
        static QuantityTensor3D _toSimSpaceScaling(
            const QuantityTensor3D &,
            const molsys::Box &
        );

        [[nodiscard]]
        const PressureTensor3D &_getPressureTensor() const;

        [[nodiscard]]
        const Pressure &_getPressure() const;

        void _setPressureTensor(const PressureTensor3D &pTensor);
        void _setPressure(const Pressure &pressure);

        [[nodiscard]]
        const ManostatSettings &_getSettings() const;

       private:
        static void _validateCandidate(molsys::Box &, const QuantityTensor3D &);
    };

}   // namespace manostat

#endif   // MANOSTAT__MANOSTAT_HPP
