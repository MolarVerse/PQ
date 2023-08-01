#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace readInput;

/**
 * @brief parses the coulomb cutoff radius
 *
 * @param lineElements
 */
void InputFileReader::parseCoulombRadius(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSimulationBox().setCoulombRadiusCutOff(stod(lineElements[2]));
}