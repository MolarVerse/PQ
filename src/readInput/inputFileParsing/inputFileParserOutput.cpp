#include "inputFileParser.hpp"

#include <cstddef>
#include <iostream>

using namespace std;
using namespace readInput;
using namespace output;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output object
 *
 * @param engine
 */
InputFileParserOutput::InputFileParserOutput(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("output_freq"), bind_front(&InputFileParserOutput::parseOutputFreq, this), false);
    addKeyword(string("output_file"), bind_front(&InputFileParserOutput::parseLogFilename, this), false);
    addKeyword(string("info_file"), bind_front(&InputFileParserOutput::parseInfoFilename, this), false);
    addKeyword(string("energy_file"), bind_front(&InputFileParserOutput::parseEnergyFilename, this), false);
    addKeyword(string("traj_file"), bind_front(&InputFileParserOutput::parseTrajectoryFilename, this), false);
    addKeyword(string("vel_file"), bind_front(&InputFileParserOutput::parseVelocityFilename, this), false);
    addKeyword(string("force_file"), bind_front(&InputFileParserOutput::parseForceFilename, this), false);
    addKeyword(string("restart_file"), bind_front(&InputFileParserOutput::parseRestartFilename, this), false);
    addKeyword(string("charge_file"), bind_front(&InputFileParserOutput::parseChargeFilename, this), false);
}

/**
 * @brief parse output frequency of simulation and set it in output statically
 *
 * @param lineElements
 */
void InputFileParserOutput::parseOutputFreq(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto outputFrequency = stoi(lineElements[2]);
    if (outputFrequency < 0)
        throw customException::InputFileException("Output frequency cannot be negative - \"" + lineElements[2] + "\" at line " +
                                                  to_string(lineNumber) + "in input file");

    Output::setOutputFrequency(static_cast<size_t>(outputFrequency));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseLogFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getLogOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseInfoFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getInfoOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseEnergyFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getEnergyOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseTrajectoryFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getXyzOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseVelocityFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getVelOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseForceFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getForceOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRestartFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getRstFileOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseChargeFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getChargeOutput().setFilename(lineElements[2]);
}