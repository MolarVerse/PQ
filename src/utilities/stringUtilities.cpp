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

#include "stringUtilities.hpp"

#include <algorithm>    // for __for_each_fn
#include <cctype>       // for isspace
#include <format>       // for format
#include <fstream>      // IWYU pragma: keep for basic_istream, ifstream
#include <functional>   // for identity
#include <ranges>   // for begin, end, operator|, views::split, views::transform
#include <sstream>       // IWYU pragma: keep for basic_stringstream
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "exceptions.hpp"   // for InputFil...

/**
 * @brief Removes comments from a line
 *
 * @details the comment char can be chosen freely - in the program ';' is used
 *
 * @param line
 * @param commentChar
 * @return std::string
 */
std::string utilities::removeComments(
    std::string            &line,
    const std::string_view &commentChar
)
{
    if (const auto commentPos = line.find(commentChar);
        commentPos != std::string::npos)
        line = line.substr(0, commentPos);
    return line;
}

/**
 * @brief get commands from a line
 *
 * @note split commands at every semicolon
 *
 * @param line
 * @return std::vector<std::string>
 *
 * @throw InputFileException if line does not end with a semicolon
 */
std::vector<std::string> utilities::getLineCommands(
    const std::string &line,
    const size_t       lineNumber
)
{
    for (auto i = static_cast<int>(line.size() - 1); i >= 0; --i)
    {
        if (';' == line[static_cast<size_t>(i)])
            break;
        else if (!bool(::isspace(line[static_cast<size_t>(i)])))
            throw customException::InputFileException(std::format(
                "Missing semicolon in input file at line {}",
                lineNumber
            ));
    }

    using std::operator""sv;
    constexpr auto delim{";"sv};
    auto           splitView =
        line | std::views::split(delim) |
        std::views::transform([](auto &&view)
                              { return std::string(view.begin(), view.end()); }
        );

    std::vector<std::string> lineCommands;
    for (auto it : splitView) lineCommands.emplace_back(it);

    return std::vector<std::string>(
        lineCommands.begin(),
        lineCommands.end() - 1
    );
}

/**
 * @brief Splits a string into a vector of strings at every whitespace
 *
 * @param line
 * @return std::vector<std::string>
 */
std::vector<std::string> utilities::splitString(const std::string &line)
{
    std::string              word;
    std::vector<std::string> lineElements = {};

    std::stringstream ss(line);

    while (ss >> word) lineElements.push_back(word);

    return lineElements;
}

/**
 * @brief returns a copy of a string all lower case
 *
 * @param myString
 * @return string
 */
std::string utilities::toLowerCopy(std::string myString)
{
    std::ranges::for_each(myString, [](char &c) { c = char(::tolower(c)); });
    return myString;
}

/**
 * @brief returns a copy of a string all lower case
 *
 * @param myString
 * @return string
 */
std::string utilities::toLowerCopy(const std::string_view myString)
{
    return toLowerCopy(std::string(myString));
}

/**
 * @brief converts the first letter of a string to upper case and the rest to
 * lower case
 *
 * @param myString
 * @return std::string
 */
std::string utilities::firstLetterToUpperCaseCopy(std::string myString)
{
    myString[0] = char(::toupper(myString[0]));

    std::ranges::for_each(
        myString | std::views::drop(1),
        [](char &c) { c = char(::tolower(c)); }
    );

    return myString;
}

/**
 * @brief checks if a file exists and can be opened
 *
 * @param filename
 * @return true if file exists and can be opened
 * @return false if file does not exist or cannot be opened
 */
bool utilities::fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

/**
 * @brief adds leading and trailing spaces to a string
 *
 * @param command
 * @param stringToReplace
 * @param lineNumber
 *
 * @throw InputFileException if equal sign is missing
 */
void utilities::addSpaces(
    std::string       &command,
    const std::string &stringToReplace,
    const size_t       lineNumber
)
{
    const auto equalSignPos = command.find(stringToReplace);
    if (equalSignPos != std::string::npos)
        command.replace(equalSignPos, 1, " " + stringToReplace + " ");

    else
        throw customException::InputFileException(std::format(
            "Missing \"{}\" in command \"{}\" in line {}",
            stringToReplace,
            command,
            lineNumber
        ));
}