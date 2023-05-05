#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

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