#include "inputFileParser.hpp"

#include <iostream>

using namespace std;
using namespace customException;
using namespace readInput;

/**
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw InputFileException if second argument is not "="
 */
void readInput::checkEqualSign(const string_view &lineElement, const size_t lineNumber)
{
    if (lineElement != "=") throw InputFileException("Invalid command at line " + to_string(lineNumber) + "in input file");
}

/**
 * @brief check if command array has at least 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less than 3 elements
 *
 * @note this function is used for commands that have an array as their third argument
 */
void readInput::checkCommandArray(const vector<string> &lineElements, const size_t lineNumber)
{
    if (lineElements.size() < 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(lineNumber) + "in input file");

    checkEqualSign(lineElements[1], lineNumber);
}

/**
 * @brief check if command array has exactly 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less or more than 3 elements
 */
void readInput::checkCommand(const vector<string> &lineElements, const size_t lineNumber)
{
    if (lineElements.size() != 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(lineNumber) + "in input file");

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
void InputFileParser::addKeyword(const string &keyword, ParseFunc parserFunc, bool required)
{
    _keywordFuncMap.try_emplace(keyword, parserFunc);
    _keywordRequiredMap.try_emplace(keyword, required);
    _keywordCountMap.try_emplace(keyword, 0);
}