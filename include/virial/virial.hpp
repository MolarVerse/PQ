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

#include "staticMatrix.hpp"   // for StaticMatrix3x3
#include "timer.hpp"          // for Timer
#include "typeAliases.hpp"

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
        std::string _virialType;   // TODO: make this an enum

        pq::tensor3D _virial;

       public:
        virtual ~Virial() = default;

        virtual std::shared_ptr<Virial> clone() const = 0;

        virtual void calculateVirial(pq::SimBox &, pq::PhysicalData &);
        virtual void intraMolecularVirialCorrection(pq::SimBox &, pq::PhysicalData &){
        };

        void setVirial(const pq::tensor3D &virial);

        [[nodiscard]] pq::tensor3D getVirial() const;
        [[nodiscard]] std::string  getVirialType() const;
    };
}   // namespace virial

#endif   // _VIRIAL_HPP_