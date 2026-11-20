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

#include "systemMemory.hpp"

#include <fstream>
#include <string>

namespace utilities
{
    /**
     * @brief memory available for new allocations without swapping
     *
     * @details reads MemAvailable from /proc/meminfo
     *
     * @return std::optional<std::size_t> bytes, or nothing if unknown, e.g.
     * not on Linux
     */
    std::optional<std::size_t> availableMemory()
    {
        constexpr std::size_t bytesPerKiB = 1024;

        std::ifstream meminfo("/proc/meminfo");
        std::string   key;
        std::size_t   kiB = 0;
        std::string   unit;

        while (meminfo >> key >> kiB >> unit)
            if (key == "MemAvailable:")
                return kiB * bytesPerKiB;

        return std::nullopt;
    }

}   // namespace utilities
