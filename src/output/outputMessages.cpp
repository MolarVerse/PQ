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

#include "outputMessages.hpp"

#include <format>    // for format
#include <sstream>   // for stringstream
#include <string>    // for string

#include "systemInfo.hpp"   // for _AUTHOR_

using namespace sysinfo;

/**
 * @brief construct header title
 *
 * @return string
 */
std::string output::header()
{
    std::stringstream header_title;

    header_title << R"(
************************************************************************
*                                                                      *
*                                                                      *
*                      88888888ba     ,ad8888ba,                       *
*                      88      "8b   d8"'    `"8b                      *
*                      88      ,8P  d8'        `8b                     *
*                      88aaaaaa8P'  88          88                     *
*                      88""""""'    88          88                     *
*                      88           Y8,    "88,,8P                     *
*                      88            Y8a.    Y88P                      *
*                      88             `"Y8888Y"Y8a                     *
*                                                                      *
*                                                                      *
************************************************************************
)";

    header_title << '\n';
    header_title << _OUTPUT_ << "Author:        " << _AUTHOR_ << '\n';
    header_title << _OUTPUT_ << "Email:         " << _EMAIL_ << '\n';

    header_title << '\n';
    header_title << _OUTPUT_ << "Testing:       " << _JOSEF_ << '\n';
    header_title << _OUTPUT_ << "               " << _ARMIN_ << '\n';
    header_title << _OUTPUT_ << "               " << _STEFAN_ << '\n';
    header_title << _OUTPUT_ << "               " << _BENJAMIN_ << '\n';

    header_title << '\n';
    header_title << _OUTPUT_ << "Version:       " << _VERSION_ << '\n';
    header_title << _OUTPUT_ << "Compile date:  " << _COMPILE_DATE_ << '\n';

    return header_title.str();
}

/**
 * @brief construct ended normally message
 *
 * @return string
 */
std::string output::endedNormally()
{
    // clang-format off
    const std::string endedNormally_message = std::format(R"(
{}For citation please refer to the ".ref" file.

*************************************************************************
*                                                                       *
*                          PQ ended normally                            *
*                                                                       *
*************************************************************************
)",
_INFO_);
    // clang-format on

    return endedNormally_message;
}

/**
 * @brief construct elapsed time message
 *
 * @param elapsedTime
 * @return string
 */
std::string output::elapsedTimeMessage(const double elapsedTime)
{
    return std::format(
        "\n\n{}Elapsed time = {:.5f} s\n",
        _OUTPUT_,
        elapsedTime
    );
}

/**
 * @brief Message to inform about starting to setup
 *
 * @param setup
 * @return std::string
 */
std::string output::setupMessage(const std::string &setup)
{
    return std::format("{}Setup of {}\n", _INFO_, setup);
}

/**
 * @brief Message to inform about completed setup
 *
 * @return std::string
 */
std::string output::setupCompletedMessage()
{
    return R"(
************************ STARTING SIMULATION ****************************
)";
}

/**
 * @brief Message to inform about reading a file
 *
 * @param file
 * @return std::string
 */
std::string output::readMessage(
    const std::string &message,
    const std::string &file
)
{
    return std::format("{}Reading {} \"{}\"\n", _INFO_, message, file);
}