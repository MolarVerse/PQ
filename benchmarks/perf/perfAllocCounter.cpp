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
#include <cstddef>
#include <cstdio>

extern "C"
{
    void *__libc_malloc(std::size_t size);
    void *__libc_calloc(std::size_t count, std::size_t size);
    void *__libc_realloc(void *pointer, std::size_t size);
    void *__libc_memalign(std::size_t alignment, std::size_t size);
}

namespace
{
    unsigned long g_calls  = 0;
    unsigned long g_bytes  = 0;
    bool          g_active = false;

    void count(std::size_t size)
    {
        if (!__atomic_load_n(&g_active, __ATOMIC_RELAXED))
            return;

        __atomic_fetch_add(&g_calls, 1UL, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_bytes, size, __ATOMIC_RELAXED);
    }

    __attribute__((destructor)) void report()
    {
        if (!__atomic_exchange_n(&g_active, false, __ATOMIC_RELAXED))
            return;

        std::array<char, 96> line{};
        const auto           length = std::snprintf(
            line.data(),
            line.size(),
            "PQ_PERF_ALLOCS %lu %lu\n",
            __atomic_load_n(&g_calls, __ATOMIC_RELAXED),
            __atomic_load_n(&g_bytes, __ATOMIC_RELAXED)
        );

        if (length <= 0)
            return;

        [[maybe_unused]] const auto written = ::write(
            STDERR_FILENO,
            line.data(),
            static_cast<std::size_t>(length)
        );
    }
}   // namespace

extern "C"
{
    void pq_perf_alloc_begin()
    {
        __atomic_store_n(&g_calls, 0UL, __ATOMIC_RELAXED);
        __atomic_store_n(&g_bytes, 0UL, __ATOMIC_RELAXED);
        __atomic_store_n(&g_active, true, __ATOMIC_RELAXED);
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
