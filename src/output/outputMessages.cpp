/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "systemInfo.hpp"   // for _AUTHOR_

#include <format>    // for format
#include <sstream>   // for stringstream
#include <string>    // for string

/**
 * @brief construct header title
 *
 * @return string
 */
std::string output::header()
{
    std::stringstream header_title;

    header_title << R"(
*************************************************************************
*                                                                       *
*                            _                                    ___   *
*          _                ( )                                 /'___)  *
*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *
*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *
*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *
*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *
*  | |                                    | |                           *
*  (_)                                    (_)                           *
*                                                                       *
*                                                                       *
*************************************************************************
)";

    header_title << '\n';
    header_title << _OUTPUT_ << "Author: " << sysinfo::_AUTHOR_ << '\n';
    header_title << _OUTPUT_ << "Email:  " << sysinfo::_EMAIL_ << '\n';

    return header_title.str();
}

/**
 * @brief construct ended normally message
 *
 * @return string
 */
std::string output::endedNormally()
{
    const std::string endedNormally_message = R"(
*************************************************************************
*                                                                       *
*                      pimd-qmcf ended normally                         *
*                                                                       *
*************************************************************************
)";

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
    return std::format("\n\n{}Elapsed time = {} s\n", _OUTPUT_, elapsedTime);
}