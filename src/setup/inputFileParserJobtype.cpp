#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief parse jobtype of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throw invalid_argument if jobtype is not recognised
 */
void InputFileReader::parseJobType(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "mm-md")
        _engine._jobType = MMMD();
    else
        throw invalid_argument("Invalid jobtype \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}