#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace readInput;

/**
 * @brief parses the coulomb cutoff radius
 *
 * @param lineElements
 */
void InputFileReader::parseRcoulomb(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSimulationBox().setRcCutOff(stod(lineElements[2]));
}