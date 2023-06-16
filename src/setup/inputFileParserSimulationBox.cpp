#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace setup;

void InputFileReader::parseRcoulomb(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSimulationBox().setRcCutOff(stod(lineElements[2]));
}