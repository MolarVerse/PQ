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

#ifndef BENCHMARKS__PERF__PERF_ALLOC_HPP
#define BENCHMARKS__PERF__PERF_ALLOC_HPP

// Marks the start of the measured work for the allocation counter
// (perfAllocCounter.cpp), which scripts/perf_gate.sh preloads. Without the
// library the weak symbol is null and nothing happens.
extern "C" void pq_perf_alloc_begin() __attribute__((weak));

inline void perfAllocBegin()
{
    if (pq_perf_alloc_begin != nullptr)
        pq_perf_alloc_begin();
}

#endif   // BENCHMARKS__PERF__PERF_ALLOC_HPP
