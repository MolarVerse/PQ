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
// pq_perf_alloc_begin() (perfAlloc.hpp) to the end of the process, and the peak
// of the heap in use (usable bytes of all live blocks, also those allocated
// before the call) from then on.
//
// Not linked into the benchmarks: scripts/perf_gate.sh preloads it with
// LD_PRELOAD in a plain run next to the callgrind run, so the instruction
// counts are not influenced. It replaces the C allocation functions (which
// operator new uses) and prints "PQ_PERF_ALLOCS <calls> <bytes> <peak>" to
// stderr at exit. The numbers are exact and do not depend on the CPU.
//
// Linux with glibc only (it calls the __libc_* allocators directly).

#include <unistd.h>

#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

// glibc exports its own allocators under these (reserved) names; they are the
// way to call the real allocator from a replaced malloc.
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C"
{
    void       *__libc_malloc(std::size_t size);
    void       *__libc_calloc(std::size_t count, std::size_t size);
    void       *__libc_realloc(void *pointer, std::size_t size);
    void       *__libc_memalign(std::size_t alignment, std::size_t size);
    void        __libc_free(void *pointer);
    std::size_t malloc_usable_size(void *pointer);
}
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)

namespace
{
    struct Counters
    {
        std::atomic<std::uint64_t> calls{0};
        std::atomic<std::uint64_t> bytes{0};
        std::atomic<std::int64_t>  live{0};
        std::atomic<std::int64_t>  peak{0};
        std::atomic<bool>          active{false};
        // every calloc of the process, also while not active
        std::atomic<std::uint64_t> callocs{0};
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

    // heap in use: every block is accounted with its usable size from the
    // start of the process, the peak only from the marker on
    void changeLive(std::int64_t delta)
    {
        auto &state = counters();

        const auto now =
            state.live.fetch_add(delta, std::memory_order_relaxed) + delta;

        if (delta <= 0 || !state.active.load(std::memory_order_relaxed))
            return;

        auto peak = state.peak.load(std::memory_order_relaxed);
        while (now > peak && !state.peak.compare_exchange_weak(
                                 peak,
                                 now,
                                 std::memory_order_relaxed
                             ))
        {
        }
    }

    std::int64_t usable(void *pointer)
    {
        return pointer == nullptr
                   ? 0
                   : static_cast<std::int64_t>(malloc_usable_size(pointer));
    }

    void *track(void *pointer)
    {
        changeLive(usable(pointer));
        return pointer;
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

    void doNothing() {}

    // glibc keeps the first 32 exit handlers (atexit / __cxa_atexit, which the
    // first use of every function-local static with a destructor registers) in
    // a static block and callocs the next block of 32 only when it is full.
    // Whether that calloc falls into the measured work therefore depends on how
    // many static destructors the process registered before the marker, so
    // adding or removing an unrelated static object would shift one allocation
    // into or out of the measurement. Registering handlers until glibc has
    // allocated a fresh block moves that allocation in front of the marker and
    // leaves 31 free slots for the measured work.
    void reserveExitHandlers()
    {
        constexpr std::size_t maxHandlers = 256;

        auto &state = counters();

        const auto before = state.callocs.load(std::memory_order_relaxed);

        for (std::size_t i = 0; i < maxHandlers; ++i)
        {
            const auto _ = std::atexit(doNothing);

            if (state.callocs.load(std::memory_order_relaxed) != before)
                return;
        }
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
        append(line, length, " ");
        append(
            line,
            length,
            static_cast<std::uint64_t>(
                state.peak.load(std::memory_order_relaxed)
            )
        );
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

        reserveExitHandlers();

        state.calls.store(0, std::memory_order_relaxed);
        state.bytes.store(0, std::memory_order_relaxed);
        state.peak.store(
            state.live.load(std::memory_order_relaxed),
            std::memory_order_relaxed
        );
        state.active.store(true, std::memory_order_relaxed);
    }

    void *malloc(std::size_t size)
    {
        count(size);
        return track(__libc_malloc(size));
    }

    void free(void *pointer)
    {
        changeLive(-usable(pointer));
        __libc_free(pointer);
    }

    void *calloc(std::size_t number, std::size_t size)
    {
        counters().callocs.fetch_add(1, std::memory_order_relaxed);
        count(number * size);
        return track(__libc_calloc(number, size));
    }

    void *realloc(void *pointer, std::size_t size)
    {
        count(size);
        changeLive(-usable(pointer));
        return track(__libc_realloc(pointer, size));
    }

    void *memalign(std::size_t alignment, std::size_t size)
    {
        count(size);
        return track(__libc_memalign(alignment, size));
    }

    void *aligned_alloc(std::size_t alignment, std::size_t size)
    {
        count(size);
        return track(__libc_memalign(alignment, size));
    }

    int posix_memalign(void **result, std::size_t alignment, std::size_t size)
    {
        count(size);
        *result = track(__libc_memalign(alignment, size));
        return *result == nullptr ? 12 : 0;
    }
}
