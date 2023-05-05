#include <iostream>

#include "inputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief parse jobtype of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if jobtype is not recognised
 */
void InputFileReader::parseJobType(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "mm-md")
        _engine._jobType = MMMD();
    else
        throw InputFileException("Invalid jobtype \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}