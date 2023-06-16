#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace setup;

void InputFileReader::parseDensity(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSimulationBox().setDensity(stod(lineElements[2]));
}