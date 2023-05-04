#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw invalid_argument if second argument is not "="
 */
void checkEqualSign(string_view lineElement, int _lineNumber)
{
    if (lineElement != "=")
        throw invalid_argument("Invalid command at line " + to_string(_lineNumber) + "in input file");
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
void checkCommandArray(const vector<string> &lineElements, int _lineNumber)
{
    if (lineElements.size() < 3)
        throw invalid_argument("Invalid number of arguments at line " + to_string(_lineNumber) + "in input file");

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
void checkCommand(const vector<string> &lineElements, int _lineNumber)
{
    if (lineElements.size() != 3)
        throw invalid_argument("Invalid number of arguments at line " + to_string(_lineNumber) + "in input file");

    checkEqualSign(lineElements[1], _lineNumber);
}

/**
 * @brief parse jobtype of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throw invalid_argument if jobtype is not recognised
 */
void InputFileReader::parseJobType(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "mm-md")
        _settings._jobType = MMMD();
    else
        throw invalid_argument("Invalid jobtype \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}

/**
 * @brief parse timestep of simulation and set it in timings
 *
 * @param lineElements
 */
void InputFileReader::parseTimestep(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _settings._timings.setTimestep(stoi(lineElements[2]));
}

/**
 * @brief parse number of steps of simulation and set it in timings
 *
 * @param lineElements
 */
void InputFileReader::parseNumberOfSteps(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _settings._timings.setNumberOfSteps(stoi(lineElements[2]));
}

/**
 * @brief parse output frequency of simulation and set it in output statically
 *
 * @param lineElements
 */
void InputFileReader::parseOutputFreq(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    Output::setOutputFreq(stoi(lineElements[2]));
}

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseStartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _settings.setStartFilename(lineElements[2]);
}

void InputFileReader::parseLogFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = LogOutput();
    output.setFilename(lineElements[2]);
    _settings._output.push_back(output);
}