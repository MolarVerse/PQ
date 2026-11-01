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

#include <dlfcn.h>

// Marks the start of the measured work for the allocation counter
// (allocCounter/perfAllocCounter.cpp), which scripts/perf_gate.sh preloads.
// The function is looked up at run time: an undefined weak symbol would be
// resolved to null at link time by some linkers (mold, which CI uses), so the
// preloaded library would never be called. Without the library nothing
// happens.
inline void perfAllocBegin()
{
    using Begin = void (*)();

    void *const symbol = dlsym(RTLD_DEFAULT, "pq_perf_alloc_begin");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto begin = reinterpret_cast<Begin>(symbol);

    if (begin != nullptr)
        begin();
}

#endif   // BENCHMARKS__PERF__PERF_ALLOC_HPP
