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

// Fixed-work end-to-end benchmark: a short MM-MD run through the real driver
// (input parsing, setup, 100 steps of 219 rigid TIP3P waters with M-Shake,
// cell list and a Berendsen thermostat, output every 10 steps). Unlike the
// kernel benchmarks this includes setup and file output, so it also notices a
// regression in how the parts fit together. The input files are in
// benchmarks/perf/data/endToEnd and are copied to a temporary directory.

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "driver.hpp"
#include "perfAlloc.hpp"

#ifndef PQ_PERF_DATA_DIR
#error "PQ_PERF_DATA_DIR must point to benchmarks/perf/data/endToEnd"
#endif

namespace fs = std::filesystem;

int main()
{
    const auto dataDir = fs::path(PQ_PERF_DATA_DIR);
    const auto workDir = fs::temp_directory_path() /
                         ("pq_perf_endToEnd_" + std::to_string(::getpid()));
    const auto startDir = fs::current_path();

    fs::create_directories(workDir);
    for (const auto &entry : fs::directory_iterator(dataDir))
        fs::copy_file(entry.path(), workDir / entry.path().filename());
    fs::current_path(workDir);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    driver::Driver::run("run.in");

    fs::current_path(startDir);
    std::error_code ignored;
    fs::remove_all(workDir, ignored);

    std::cout << "done\n";
    return 0;
}
