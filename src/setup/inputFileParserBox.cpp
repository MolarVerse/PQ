#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

void InputFileReader::parseDensity(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._simulationBox._box.setDensity(stod(lineElements[2]));
}