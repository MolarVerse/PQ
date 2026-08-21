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

#ifndef _TIMER_ID_HPP_
#define _TIMER_ID_HPP_

#include <cstdint>
#include <mstd/enum.hpp>
#include <string>

#define TIMER_ID_LIST(X)   \
    X(Simulation, 0)       \
    X(DefaultTimings)      \
    X(EngineOutput)        \
    X(Constraints)         \
    X(CellList)            \
    X(PhysicalData)        \
    X(Integrator)          \
    X(Thermostat)          \
    X(Manostat)            \
    X(Output)              \
    X(Potential)           \
    X(IntraNonBonded)      \
    X(Virial)              \
    X(ResetKinetics)       \
    X(WaterIntraPotential) \
    X(WaterInterPotential) \
    X(QMEngine)            \
    X(Setup)

MSTD_ENUM(TimerId, std::uint8_t, TIMER_ID_LIST);

std::string toString(TimerId id);

#endif   // _TIMER_ID_HPP_
