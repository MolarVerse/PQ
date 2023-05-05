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

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseMoldescriptorFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._settings.setMoldescriptorFilename(lineElements[2]);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseGuffPath(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._settings.setGuffPath(lineElements[2]);
}