#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseStartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._settings.setStartFilename(lineElements[2]);
}