#include "inputFileReader.hpp"

#include <iostream>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseStartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setStartFilename(lineElements[2]);
}

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseMoldescriptorFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setMoldescriptorFilename(lineElements[2]);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseGuffPath(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setGuffPath(lineElements[2]);
}

/**
 * @brief parse guff dat file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileReader::parseGuffDatFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setGuffDatFilename(lineElements[2]);
}

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
        _engine.getSettings().setJobtype("MMMD");
    else
        throw InputFileException("Invalid jobtype \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
}