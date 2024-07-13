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

#include "inputFileParser.hpp"

#include <format>        // for format
#include <string_view>   // for string_view

#include "exceptions.hpp"        // for InputFileException
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;
using namespace customException;
using namespace utilities;

/**
 * @brief check if parameter is "="
 *
 * @param view
 * @param _lineNumber
 *
 * @throw InputFileException if argument is not "="
 */
void input::checkEqualSign(
    const std::string_view &view,
    const size_t            lineNumber
)
{
    if (view != "=")
        throw InputFileException(
            std::format("Invalid command at line {} in input file", lineNumber)
        );
}

/**
 * @brief check if command array has at least 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less than 3
 * elements
 *
 * @note this function is used for commands that have an array as their third
 * argument
 */
void input::checkCommandArray(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    if (lineElements.size() < 3)
        throw InputFileException(std::format(
            "Invalid number of arguments at line {} in input file",
            lineNumber
        ));

    checkEqualSign(lineElements[1], lineNumber);
}

/**
 * @brief check if command array has exactly 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less or more
 * than 3 elements
 */
void input::checkCommand(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    if (lineElements.size() != 3)
        throw InputFileException(std::format(
            "Invalid number of arguments at line {} in input file",
            lineNumber
        ));

    checkEqualSign(lineElements[1], lineNumber);
}

/**
 * @brief add keyword to different keyword maps
 *
 * @param keyword
 * @param parserFunc
 * @param count
 * @param required
 *
 * @details
 *
 *  parserFunc is a function pointer to a parsing function
 *  count is the number of keywords found in the inputfile
 *  required is a boolean that indicates if the keyword is required
 *
 */
void InputFileParser::addKeyword(
    const std::string &keyword,
    pq::ParseFunc      parserFunc,
    bool               required
)
{
    const auto keywordLowerCase = toLowerCopy(keyword);
    _keywordFuncMap.try_emplace(keywordLowerCase, parserFunc);
    _keywordRequiredMap.try_emplace(keywordLowerCase, required);
    _keywordCountMap.try_emplace(keywordLowerCase, 0);
}

/**
 * @brief get the keyword function map
 *
 * @return the keyword function map
 */
std::map<std::string, pq::ParseFunc> InputFileParser::getKeywordFuncMap() const
{
    return _keywordFuncMap;
}

/**
 * @brief get the keyword required map
 *
 * @return the keyword required map
 */
std::map<std::string, bool> InputFileParser::getKeywordRequiredMap() const
{
    return _keywordRequiredMap;
}

/**
 * @brief get the keyword count map
 *
 * @return the keyword count map
 */
std::map<std::string, int> InputFileParser::getKeywordCountMap() const
{
    return _keywordCountMap;
}