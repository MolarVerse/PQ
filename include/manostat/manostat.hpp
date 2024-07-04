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

#ifndef _MANOSTAT_HPP_

#define _MANOSTAT_HPP_

#include "manostatSettings.hpp"
#include "staticMatrix3x3.hpp"   // for tensor3D
#include "timer.hpp"             // for Timer
#include "typeAliases.hpp"

namespace manostat
{
    /**
     * @class Manostat
     *
     * @brief Manostat is a base class for all manostats
     *
     */
    class Manostat : public timings::Timer
    {
       protected:
        pq::tensor3D _pressureTensor = {0.0};
        double       _pressure;
        double       _targetPressure;   // no default value, must be set

       public:
        explicit Manostat(const double targetPressure);
        Manostat()          = default;
        virtual ~Manostat() = default;

        void         calculatePressure(const pq::SimBox &, pq::PhysicalData &);
        virtual void applyManostat(pq::SimBox &, pq::PhysicalData &);

        virtual pq::ManostatType getManostatType() const;
        virtual pq::Isotropy     getIsotropy() const;
    };

}   // namespace manostat

#endif   // _MANOSTAT_HPP_