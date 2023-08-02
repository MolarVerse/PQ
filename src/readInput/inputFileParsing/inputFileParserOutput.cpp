#include "inputFileReader.hpp"

#include <cstddef>
#include <iostream>

using namespace std;
using namespace readInput;
using namespace output;

/**
 * @brief parse output frequency of simulation and set it in output statically
 *
 * @param lineElements
 */
void InputFileReader::parseOutputFreq(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    const auto outputFrequency = stoi(lineElements[2]);
    if (outputFrequency < 0)
        throw customException::InputFileException("Output frequency cannot be negative - \"" + lineElements[2] + "\" at line " +
                                                  to_string(_lineNumber) + "in input file");

    Output::setOutputFrequency(static_cast<size_t>(outputFrequency));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseLogFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getLogOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseInfoFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getInfoOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseEnergyFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getEnergyOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseTrajectoryFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getXyzOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseVelocityFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getVelOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseForceFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getForceOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseRestartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getRstFileOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseChargeFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getChargeOutput().setFilename(lineElements[2]);
}