#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace readInput;

/**
 * @brief parse density of simulation and set it in simulation box
 *
 * @param lineElements
 */
void InputFileReader::parseDensity(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSimulationBox().setDensity(stod(lineElements[2]));
}