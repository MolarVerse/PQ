#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace readInput;
using namespace virial;
using namespace customException;

/**
 * @brief parses virial command
 *
 * @param lineElements
 */
void InputFileReader::parseVirial(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "molecular")
    {
        _engine.makeVirial(VirialMolecular());
    }
    else if (lineElements[2] == "atomic")
    {
        _engine.makeVirial(VirialAtomic());
        _engine.getPhysicalData().changeKineticVirialToAtomic();
    }
    else
        throw InputFileException("Invalid virial setting \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
}