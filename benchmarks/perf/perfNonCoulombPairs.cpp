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

// Fixed-work micro-benchmark of the non-Coulomb pair kernels
// (Lennard-Jones, Buckingham, Morse).

#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "buckinghamPair.hpp"
#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    auto lj    = potential::LennardJonesPair(9.0, 2.0, 3.0);
    auto buck  = potential::BuckinghamPair(9.0, 1.0, 0.3, 2.0);
    auto morse = potential::MorsePair(9.0, 1.0, 2.0, 1.5);

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        const double distance =
            1.0 + static_cast<double>(i & 255) * 0.03;   // within cutoff

        const auto [eLj, fLj]       = lj.calculate(distance);
        const auto [eBuck, fBuck]   = buck.calculate(distance);
        const auto [eMorse, fMorse] = morse.calculate(distance);

        sink += eLj + fLj + eBuck + fBuck + eMorse + fMorse;
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
