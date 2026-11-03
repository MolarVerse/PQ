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

// Scaling benchmark: output of one frame of 400 water molecules. See
// perfOutputWriters.hpp.

#include <cstddef>
#include <cstdint>

#include "perfOutputWriters.hpp"

static constexpr std::size_t   MOLECULES  = 400;
static constexpr std::uint64_t ITERATIONS = 2;

int main() { return benchOutputWriters::run(MOLECULES, ITERATIONS); }
