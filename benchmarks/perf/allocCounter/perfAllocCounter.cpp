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

// Counts the heap allocations of a perf benchmark, from the call of
// pq_perf_alloc_begin() (perfAlloc.hpp) to the end of the process.
//
// Not linked into the benchmarks: scripts/perf_gate.sh preloads it with
// LD_PRELOAD in a plain run next to the callgrind run, so the instruction
// counts are not influenced. It replaces the C allocation functions (which
// operator new uses) and prints "PQ_PERF_ALLOCS <calls> <bytes>" to stderr at
// exit. Allocation counts are exact and do not depend on the CPU.
//
// Linux with glibc only (it calls the __libc_* allocators directly).

#include <unistd.h>

#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string_view>

// glibc exports its own allocators under these (reserved) names; they are the
// way to call the real allocator from a replaced malloc.
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C"
{
    void *__libc_malloc(std::size_t size);
    void *__libc_calloc(std::size_t count, std::size_t size);
    void *__libc_realloc(void *pointer, std::size_t size);
    void *__libc_memalign(std::size_t alignment, std::size_t size);
}
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)

namespace
{
    struct Counters
    {
        std::atomic<std::uint64_t> calls{0};
        std::atomic<std::uint64_t> bytes{0};
        std::atomic<bool>          active{false};
    };

    // constant-initialised, so it is usable from the first malloc of the
    // process on (before any static initialiser ran)
    Counters &counters()
    {
        static Counters value;
        return value;
    }

    void count(std::size_t size)
    {
        auto &state = counters();

        if (!state.active.load(std::memory_order_relaxed))
            return;

        state.calls.fetch_add(1, std::memory_order_relaxed);
        state.bytes.fetch_add(size, std::memory_order_relaxed);
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
        auto &state = counters();

        if (!state.active.exchange(false, std::memory_order_relaxed))
            return;

        std::array<char, 96> line{};
        std::size_t          length = 0;

        append(line, length, "PQ_PERF_ALLOCS ");
        append(line, length, state.calls.load(std::memory_order_relaxed));
        append(line, length, " ");
        append(line, length, state.bytes.load(std::memory_order_relaxed));
        append(line, length, "\n");

        [[maybe_unused]] const auto written =
            ::write(STDERR_FILENO, line.data(), length);
    }
}   // namespace

extern "C"
{
    void pq_perf_alloc_begin()
    {
        auto &state = counters();

        state.calls.store(0, std::memory_order_relaxed);
        state.bytes.store(0, std::memory_order_relaxed);
        state.active.store(true, std::memory_order_relaxed);
    }

    void *malloc(std::size_t size)
    {
        count(size);
        return __libc_malloc(size);
    }

    void *calloc(std::size_t number, std::size_t size)
    {
        count(number * size);
        return __libc_calloc(number, size);
    }

    void *realloc(void *pointer, std::size_t size)
    {
        count(size);
        return __libc_realloc(pointer, size);
    }

    void *memalign(std::size_t alignment, std::size_t size)
    {
        count(size);
        return __libc_memalign(alignment, size);
    }

    void *aligned_alloc(std::size_t alignment, std::size_t size)
    {
        count(size);
        return __libc_memalign(alignment, size);
    }

    int posix_memalign(void **result, std::size_t alignment, std::size_t size)
    {
        count(size);
        *result = __libc_memalign(alignment, size);
        return *result == nullptr ? 12 : 0;
    }
}
