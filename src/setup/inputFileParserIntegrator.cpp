#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

void InputFileReader::parseIntegrator(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "v-verlet")
        _engine._integrator = VelocityVerlet();
    else
        throw InputFileException("Invalid integrator \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}