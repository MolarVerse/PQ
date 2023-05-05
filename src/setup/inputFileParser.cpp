#include <iostream>

#include "inputFileReader.hpp"

using namespace std;

/**
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw invalid_argument if second argument is not "="
 */
void Setup::InputFileReader::checkEqualSign(string_view lineElement, int _lineNumber)
{
    if (lineElement != "=")
        throw InputFileException("Invalid command at line " + to_string(_lineNumber) + "in input file");
}

/**
 * @brief check if command array has at least 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw invalid_argument if command array has less than 3 elements
 *
 * @note this function is used for commands that have an array as their third argument
 */
void Setup::InputFileReader::checkCommandArray(const vector<string> &lineElements, int _lineNumber)
{
    if (lineElements.size() < 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(_lineNumber) + "in input file");

    checkEqualSign(lineElements[1], _lineNumber);
}

/**
 * @brief check if command array has exactly 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw invalid_argument if command array has less or more than 3 elements
 */
void Setup::InputFileReader::checkCommand(const vector<string> &lineElements, int _lineNumber)
{
    if (lineElements.size() != 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(_lineNumber) + "in input file");

    checkEqualSign(lineElements[1], _lineNumber);
}