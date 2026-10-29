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

// Scaling benchmark: cell list update and inter-water forces for 512 water
// molecules (8^3 grid). See perfWaterScaling.hpp.

#include <cstdint>

#include "perfWaterScaling.hpp"

static constexpr std::size_t   PER_SIDE   = 8;
static constexpr std::uint64_t ITERATIONS = 10;

int main() { return benchWaterScaling::run(PER_SIDE, ITERATIONS); }
