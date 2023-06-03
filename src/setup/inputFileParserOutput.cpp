#include <iostream>

#include "inputFileReader.hpp"

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief parse output frequency of simulation and set it in output statically
 *
 * @param lineElements
 */
void InputFileReader::parseOutputFreq(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    Output::setOutputFrequency(stoi(lineElements[2]));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseLogFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._logOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseInfoFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._infoOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseEnergyFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._energyOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseTrajectoryFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._xyzOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseVelocityFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._velOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseForceFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._forceOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseRestartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._rstFileOutput->setFilename(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseChargeFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._chargeOutput->setFilename(lineElements[2]);
}