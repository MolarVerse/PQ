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

#include "parserUtils.hpp"

#include "exceptions.hpp"

namespace input
{
    /**
     * @brief check if parameter is "="
     *
     * @param view
     * @param lineNumber
     *
     * @throw exc::InputFileException if argument is not "="
     */
    void checkEqualSign(const std::string_view &view, size_t lineNumber)
    {
        if (view != "=")
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid command at line {} in input file",
                    lineNumber
                )
            );
        }
    }

    /**
     * @brief check if command array has at least 3 elements
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if command array has less than 3
     * elements
     *
     * @note this function is used for commands that have an array as their
     * third argument
     */
    void checkCommandArray(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        if (lineElements.size() < 3)
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid number of arguments at line {} in input file",
                    lineNumber
                )
            );
        }

        checkEqualSign(lineElements[1], lineNumber);
    }

    /**
     * @brief check if command array has exactly 3 elements
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if command array has less or more
     * than 3 elements
     */
    void checkCommand(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        if (lineElements.size() != 3)
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid number of arguments at line {} in input file",
                    lineNumber
                )
            );
        }

        checkEqualSign(lineElements[1], lineNumber);
    }
}   // namespace input
