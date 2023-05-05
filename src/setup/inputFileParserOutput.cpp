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
    Output::setOutputFreq(stoi(lineElements[2]));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseLogFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = LogOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseInfoFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = InfoOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseEnergyFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = EnergyOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseTrajectoryFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = TrajectoryOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseVelocityFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = TrajectoryOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseRestartFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = RstFileOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileReader::parseChargeFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto output = ChargeOutput();
    output.setFilename(lineElements[2]);
    _engine._output.push_back(output);
}