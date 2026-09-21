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
using namespace exc;
using namespace utilities;

/**
 * @brief add keyword to different keyword maps
 *
 * @param keyword
 * @param parserFunc
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
    ParseFunc          parserFunc,
    bool               required
)
{
    const auto keywordLowerCase = toLowerAndReplaceDashesCopy(keyword);
    _keywordFuncMap.try_emplace(keywordLowerCase, parserFunc);
    _keywordRequiredMap.try_emplace(keywordLowerCase, required);
    _keywordCountMap.try_emplace(keywordLowerCase, 0);
}

/**
 * @brief get the keyword function map
 *
 * @return the keyword function map
 */
std::map<std::string, InputFileParser::ParseFunc> InputFileParser::
    getKeywordFuncMap() const
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

/**
 * @brief clear all keyword maps
 *
 * @details
 *
 * This function clears all the keyword maps, effectively resetting the parser
 * state.
 */
void InputFileParser::_clear()
{
    _keywordFuncMap.clear();
    _keywordRequiredMap.clear();
    _keywordCountMap.clear();

    _registry.clearValues();
}

/**
 * @brief get the input registry
 *
 * @return the input registry
 */
InputRegistry &InputFileParser::_getRegistry() { return _registry; }
