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

#include <ranges>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>

#include <vector>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>

#include <cstdint>
#include <vector>
#endif

namespace
{
#if defined(_WIN32)
    constexpr char _pathSeparator = ';';
#else
    constexpr char _pathSeparator = ':';
#endif
}   // namespace

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

/**
 * @brief true if path is a regular file that someone may execute
 *
 * @param path candidate file
 */
bool utilities::isExecutableFile(const std::filesystem::path &path)
{
    auto       error  = std::error_code{};
    const auto status = std::filesystem::status(path, error);

    if (error || !std::filesystem::is_regular_file(status))
    {
        return false;
    }

    constexpr auto anyExec = std::filesystem::perms::owner_exec |
                             std::filesystem::perms::group_exec |
                             std::filesystem::perms::others_exec;

    return (status.permissions() & anyExec) != std::filesystem::perms::none;
}

/**
 * @brief first executable programName in the directories of searchPath, like
 * the shell resolves a command from PATH
 *
 * @param searchPath directory list in the format of PATH
 * @param programName file name to look for
 * @return std::nullopt if no directory holds an executable programName
 */
std::optional<std::filesystem::path> utilities::findInSearchPath(
    std::string_view             searchPath,
    const std::filesystem::path &programName
)
{
    for (const auto entry : searchPath | std::views::split(_pathSeparator))
    {
        const auto directory = std::string_view(entry.begin(), entry.end());
        if (directory.empty())
        {
            continue;
        }

        auto candidate = std::filesystem::path(directory) / programName;
        if (isExecutableFile(candidate))
        {
            return candidate;
        }
    }

    return std::nullopt;
}
