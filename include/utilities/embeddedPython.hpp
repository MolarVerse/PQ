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

#ifndef UTILITIES__EMBEDDED_PYTHON_HPP
#define UTILITIES__EMBEDDED_PYTHON_HPP

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace utilities
{
    [[nodiscard]] std::string pythonProgramName();

    [[nodiscard]] std::optional<std::filesystem::path> findPythonProgram(
        std::string_view             searchPath,
        std::string_view             programName,
        const std::filesystem::path &buildPython
    );

    void startPython();
    void stopPython();
}   // namespace utilities

#endif   // UTILITIES__EMBEDDED_PYTHON_HPP
