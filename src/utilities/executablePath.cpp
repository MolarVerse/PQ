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

#include "executablePath.hpp"

#if defined(_WIN32)
#include <vector>
#include <windows.h>
#elif defined(__APPLE__)
#include <cstdint>
#include <vector>
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <system_error>
#endif

std::filesystem::path utilities::executablePath()
{
#if defined(_WIN32)
    auto buffer = std::vector<wchar_t>(1024);
    while (true)
    {
        const auto length = GetModuleFileNameW(
            nullptr,
            buffer.data(),
            static_cast<DWORD>(buffer.size())
        );
        if (length == 0)
            break;
        if (length < buffer.size() - 1)
            return std::filesystem::weakly_canonical(
                std::filesystem::path(buffer.data())
            );
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    auto size = uint32_t{0};
    _NSGetExecutablePath(nullptr, &size);
    auto buffer = std::vector<char>(size);
    if (_NSGetExecutablePath(buffer.data(), &size) == 0)
        return std::filesystem::weakly_canonical(buffer.data());
#elif defined(__linux__)
    auto error = std::error_code{};
    return std::filesystem::canonical("/proc/self/exe", error);
#endif

    return {};
}

std::filesystem::path utilities::installedDataPath(
    const std::filesystem::path &relativePath
)
{
    const auto executable = executablePath();
    if (executable.empty())
        return {};

    return executable.parent_path().parent_path() / "share" / "PQ" /
           relativePath;
}
