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

#ifndef _STRING_UTILITIES_HPP_

#define _STRING_UTILITIES_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

/**
 * @brief utilities is a namespace for all utility functions
 */
namespace utilities
{
    std::string removeComments(std::string &line, const std::string_view &commentChar);

    std::vector<std::string> getLineCommands(const std::string &line, const size_t lineNumber);

    std::vector<std::string> splitString(const std::string &);

    std::string toLowerCopy(std::string);
    std::string toLowerCopy(std::string_view);
    std::string firstLetterToUpperCaseCopy(std::string);

    bool fileExists(const std::string &);

}   // namespace utilities

#endif   // _STRING_UTILITIES_HPP_