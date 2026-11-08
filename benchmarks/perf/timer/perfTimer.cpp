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

// Measures the CPU time of the measured loop of a perf benchmark, from the call
// of pq_perf_alloc_begin() (perfAlloc.hpp, the same marker as the allocation
// counter) to the end of the process. Process startup and the setup of the
// benchmark are not part of it.
//
// Not linked into the benchmarks: scripts/perf_timing.py preloads it with
// LD_PRELOAD in a plain run. Unlike the allocation counter it does not replace
// any other function, so it does not disturb the timing. It prints
// "PQ_PERF_TIME <cpu ns> <wall ns>" to stderr at exit.

#include <unistd.h>

#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <string_view>

namespace
{
    struct Stamp
    {
        std::atomic<std::uint64_t> cpu{0};
        std::atomic<std::uint64_t> wall{0};
        std::atomic<bool>          active{false};
    };

    Stamp &stamp()
    {
        static Stamp value;
        return value;
    }

    std::uint64_t now(const clockid_t clock)
    {
        constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;

        std::timespec               time{};
        [[maybe_unused]] const auto status = ::clock_gettime(clock, &time);

        return (static_cast<std::uint64_t>(time.tv_sec) * nanosecondsPerSecond
               ) +
               static_cast<std::uint64_t>(time.tv_nsec);
    }

    void append(
        std::array<char, 96> &line,
        std::size_t          &length,
        std::string_view      text
    )
    {
        for (const auto character : text) line.at(length++) = character;
    }

    void append(
        std::array<char, 96> &line,
        std::size_t          &length,
        std::uint64_t         number
    )
    {
        const auto result = std::to_chars(
            line.data() + length,
            line.data() + line.size(),
            number
        );
        length = static_cast<std::size_t>(result.ptr - line.data());
    }

    __attribute__((destructor)) void report()
    {
        auto &state = stamp();

        if (!state.active.exchange(false, std::memory_order_relaxed))
            return;

        const auto cpu  = now(CLOCK_PROCESS_CPUTIME_ID);
        const auto wall = now(CLOCK_MONOTONIC);

        std::array<char, 96> line{};
        std::size_t          length = 0;

        append(line, length, "PQ_PERF_TIME ");
        append(line, length, cpu - state.cpu.load(std::memory_order_relaxed));
        append(line, length, " ");
        append(line, length, wall - state.wall.load(std::memory_order_relaxed));
        append(line, length, "\n");

        [[maybe_unused]] const auto written =
            ::write(STDERR_FILENO, line.data(), length);
    }
}   // namespace

extern "C" void pq_perf_alloc_begin()
{
    auto &state = stamp();

    state.cpu.store(now(CLOCK_PROCESS_CPUTIME_ID), std::memory_order_relaxed);
    state.wall.store(now(CLOCK_MONOTONIC), std::memory_order_relaxed);
    state.active.store(true, std::memory_order_relaxed);
}
