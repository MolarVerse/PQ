#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief Parse the integrator used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseIntegrator(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "v-verlet")
        _engine._integrator = make_unique<VelocityVerlet>();
    else
        throw InputFileException("Invalid integrator \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}