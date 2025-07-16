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

#ifndef _MONTE_CARLO_MANOSTAT_HPP_

#define _MONTE_CARLO_MANOSTAT_HPP_

#include "manostat.hpp"                // for Manostat
#include "randomNumberGenerator.hpp"   // for RandomNumberGenerator
#include "typeAliases.hpp"             // for PhysicalData, SimulationBox

namespace manostat
{
    class MonteCarloManostat : public Manostat
    {
       private:
        randomNumberGenerator::RandomNumberGenerator _randomNumberGenerator{};

       public:
        explicit MonteCarloManostat() = default;

        void applyManostat(pq::SimBox &, pq::PhysicalData &) override {}
    };
}   // namespace manostat

#endif   // _MONTE_CARLO_MANOSTAT_HPP_
