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

#ifndef _VIRIAL_HPP_

#define _VIRIAL_HPP_

#include <string>   // for string

#include "staticMatrix3x3.hpp"   // for StaticMatrix3x3
#include "timer.hpp"             // for Timer

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

/**
 * @namespace virial
 *
 * @brief Namespace for virial calculation
 */
namespace virial
{
    /**
     * @class Virial
     *
     * @brief Base class for virial calculation
     *
     * @details implements virial calculation, which is valid for both atomic
     * and molecular systems
     */
    class Virial : public timings::Timer
    {
       protected:
        std::string _virialType;

        linearAlgebra::tensor3D _virial;

       public:
        virtual ~Virial() = default;

        virtual std::shared_ptr<Virial> clone() const = 0;

        virtual void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void intraMolecularVirialCorrection(simulationBox::SimulationBox &, physicalData::PhysicalData &) {
        };

        void setVirial(const linearAlgebra::tensor3D &virial)
        {
            _virial = virial;
        }

        [[nodiscard]] linearAlgebra::tensor3D getVirial() const
        {
            return _virial;
        }
        [[nodiscard]] std::string getVirialType() const { return _virialType; }
    };

    /**
     * @class VirialMolecular
     *
     * @brief Class for virial calculation of molecular systems
     *
     * @details overrides calculateVirial() function to include intra-molecular
     * virial correction
     */
    class VirialMolecular : public Virial
    {
       public:
        VirialMolecular() : Virial() { _virialType = "molecular"; }

        std::shared_ptr<Virial> clone() const override;

        void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &)
            override;
        void intraMolecularVirialCorrection(simulationBox::SimulationBox &, physicalData::PhysicalData &)
            override;
    };

    /**
     * @class VirialAtomic
     *
     * @brief Class for virial calculation of atomic systems
     *
     * @details dummy class for atomic systems, since no virial correction is
     * needed
     *
     */
    class VirialAtomic : public Virial
    {
       public:
        VirialAtomic() : Virial() { _virialType = "atomic"; }

        std::shared_ptr<Virial> clone() const override;
    };

}   // namespace virial

#endif   // _VIRIAL_HPP_