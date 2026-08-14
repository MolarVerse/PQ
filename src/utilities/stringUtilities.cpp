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
#include <cmath>        // for isfinite
#include <cstdint>      // for uint_fast32_t and UINT32_MAX
#include <filesystem>   // for is_regular_file
#include <format>       // for format
#include <fstream>
#include <limits>
#include <ranges>   // for begin, end, operator|, views::split, views::transform
#include <sstream>
#include <stdexcept>     // for out_of_range and invalid_argument
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "exceptions.hpp"

using namespace customException;

using std::views::split;
using std::views::transform;

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
    const auto commentPos = line.find(commentChar);

    if (commentPos != std::string::npos)
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

        else if (!static_cast<bool>(::isspace(line[static_cast<size_t>(i)])))
        {
            throw InputFileException(
                std::format(
                    "Missing semicolon in input file at line {}",
                    lineNumber
                )
            );
        }
    }

    using std::    operator""sv;
    constexpr auto delim{";"sv};

    auto transformView = [](auto &&view)
    { return std::string(view.begin(), view.end()); };

    auto splitView = line | split(delim) | transform(transformView);

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
    std::ranges::for_each(
        myString,
        [](char &c) { c = static_cast<char>(::tolower(c)); }
    );
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
 * @brief returns a copy of a string all lower case and with '-' replaced by '_'
 *
 * @param myString
 * @return string
 */
std::string utilities::toLowerAndReplaceDashesCopy(std::string myString)
{
    for (char &c : myString)
    {
        c = static_cast<char>(::tolower(c));
        if (c == '-')
            c = '_';
    }
    return myString;
}

/**
 * @brief returns a copy of a string all lower case and with '-' replaced by '_'
 *
 * @param myString
 * @return string
 */
std::string utilities::toLowerAndReplaceDashesCopy(
    const std::string_view myString
)
{
    return toLowerAndReplaceDashesCopy(std::string(myString));
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
    myString[0] = static_cast<char>(::toupper(myString[0]));

    std::ranges::for_each(
        myString | std::views::drop(1),
        [](char &c) { c = static_cast<char>(::tolower(c)); }
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
    if (!std::filesystem::is_regular_file(filename))
        return false;

    std::ifstream file(filename);
    return file.good();
}

/**
 * @brief checks if the value of a input file keyword can be interpreted as true
 * or false
 *
 * @param lineElements
 * @return true if "on", "yes", "true"
 * @return false if "off", "no", "false"
 *
 * @throw InputFileException if none of these strings is matched
 */
bool utilities::keywordToBool(const std::vector<std::string> &lineElements)
{
    const auto option = toLowerCopy(lineElements[2]);

    if ("on" == option || "yes" == option || "true" == option)
        return true;

    else if ("off" == option || "no" == option || "false" == option)
        return false;

    else
    {
        throw InputFileException(
            std::format(
                "Invalid boolean option \"{}\" for keyword \"{}\" in input "
                "file.\n"
                "Possible values are: on, yes, true, off, no, false.",
                lineElements[2],
                lineElements[0]
            )
        );
    }
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
    {
        throw customException::InputFileException(
            std::format(
                "Missing \"{}\" in command \"{}\" in line {}",
                stringToReplace,
                command,
                lineNumber
            )
        );
    }
}

/**
 * @brief converts a string to an uint_fast32_t
 *
 * @param str
 *
 * @throw invalid_argument if the string is not valid for conversion to
 * uint_fast32_t
 * @throw out_of_range if number to be converted is negative or greater than an
 * uint32
 */
std::uint_fast32_t utilities::stringToUintFast32t(const std::string &str)
{
    if (str.empty())
        throw std::invalid_argument(
            "Cannot convert empty string to unsigned integer"
        );

    size_t startPos = 0;
    if ((str[0] == '+' || str[0] == '-') && str.length() > 1)
        startPos = 1;

    for (size_t i = startPos; i < str.length(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(str[i])))
        {
            throw std::invalid_argument(
                std::format(
                    "String \"{}\" is not a valid unsigned integer",
                    str
                )
            );
        }
    }

    std::int64_t   valueLL{std::stoll(str)};
    constexpr auto maxValue = static_cast<std::int64_t>(UINT32_MAX);

    if (valueLL < 0 || valueLL > maxValue)
    {
        throw std::out_of_range(
            std::format(
                "The number has to be an integer between \"0\" and \"{}\" "
                "(inclusive)",
                maxValue
            )
        );
    }

    return static_cast<std::uint_fast32_t>(valueLL);
}

/**
 * @brief converts a string to an unsigned long long int
 *
 * @param str
 *
 * @throw invalid_argument if the string is not valid for conversion to
 * unsigned long long int
 * @throw out_of_range if number to be converted is negative or greater than an
 * unsigned long long int
 */
std::uint64_t utilities::stringToULL(const std::string &str)
{
    if (str.empty())
        throw std::invalid_argument(
            "Cannot convert empty string to unsigned long long"
        );

    const auto maxValue = std::numeric_limits<std::uint64_t>::max();

    if (str[0] == '-')
    {
        throw std::out_of_range(
            std::format(
                "The number has to be an integer between \"0\" and "
                "\"{}\" (inclusive)",
                maxValue
            )
        );
    }

    size_t startPos = (str[0] == '+') ? 1 : 0;
    if (startPos == str.length())
        throw std::invalid_argument(
            std::format("String \"{}\" is not a valid unsigned long long", str)
        );

    for (size_t i = startPos; i < str.length(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(str[i])))
        {
            throw std::invalid_argument(
                std::format(
                    "String \"{}\" is not a valid unsigned long long",
                    str
                )
            );
        }
    }

    try
    {
        return std::stoull(str);
    }
    catch (const std::out_of_range &)
    {
        throw std::out_of_range(
            std::format(
                "The number has to be an integer between \"0\" and \"{}\" "
                "(inclusive)",
                maxValue
            )
        );
    }
}

/**
 * @brief converts a complete string to an int
 *
 * @param str
 *
 * @throw invalid_argument if the complete string is not a valid integer
 * @throw out_of_range if number is out of range for an int
 */
int utilities::stringToInt(const std::string &str)
{
    size_t parsedCharacters{};
    int    value{};

    try
    {
        value = std::stoi(str, &parsedCharacters);
    }
    catch (const std::invalid_argument &)
    {
        throw std::invalid_argument(
            std::format("Invalid integer value '{}' encountered", str)
        );
    }
    catch (const std::out_of_range &)
    {
        throw std::out_of_range(
            std::format(
                "Integer value '{}' exceeds the representable range for an int",
                str
            )
        );
    }

    if (parsedCharacters != str.size())
        throw std::invalid_argument(
            std::format("Invalid integer value '{}' encountered", str)
        );

    return value;
}

/**
 * @brief converts a string to a non-Nan and non-Inf double
 *
 * @param str
 *
 * @throw invalid_argument if the string is not valid for conversion to
 * double
 * @throw out_of_range if number is out of range for a double
 */
double utilities::stringToFiniteDouble(const std::string &str)
{
    size_t parsedCharacters{};
    double value{};

    try
    {
        value = std::stod(str, &parsedCharacters);
    }
    catch (const std::invalid_argument &)
    {
        throw std::invalid_argument(
            std::format("Invalid floating-point value '{}' encountered", str)
        );
    }
    catch (const std::out_of_range &)
    {
        throw std::out_of_range(
            std::format(
                "Floating-point value '{}' exceeds the representable range "
                "for "
                "a double",
                str
            )
        );
    }

    if (parsedCharacters != str.size() || !std::isfinite(value))
        throw std::invalid_argument(
            std::format("Invalid floating-point value '{}' encountered", str)
        );

    return value;
}
