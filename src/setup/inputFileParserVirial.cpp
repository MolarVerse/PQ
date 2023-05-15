#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

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
        // default
    }
    else if (lineElements[2] == "atomic")
    {
        // TODO: implement
    }
    else
        throw InputFileException("Invalid virial setting \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}