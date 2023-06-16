#include "inputFileReader.hpp"

#include <iostream>

using namespace std;

/**
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw InputFileException if second argument is not "="
 */
void setup::checkEqualSign(const string_view &lineElement, const size_t lineNumber)
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
void setup::checkCommandArray(const vector<string> &lineElements, const size_t lineNumber)
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
void setup::checkCommand(const vector<string> &lineElements, const size_t lineNumber)
{
    if (lineElements.size() != 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(lineNumber) + "in input file");

    checkEqualSign(lineElements[1], lineNumber);
}