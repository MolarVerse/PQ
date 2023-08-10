#include "inputFileParserTimings.hpp"

#include <iostream>

using namespace std;
using namespace readInput;

/**
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw InputFileException if second argument is not "="
 */
InputFileParserTimings::InputFileParserTimings(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("timestep"), bind_front(&InputFileParserTimings::parseTimeStep, this), true);
    addKeyword(string("nstep"), bind_front(&InputFileParserTimings::parseNumberOfSteps, this), true);
}

/**
 * @brief parse timestep of simulation and set it in timings
 *
 * @param lineElements
 */
void InputFileParserTimings::parseTimeStep(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getTimings().setTimestep(stod(lineElements[2]));
}

/**
 * @brief parse number of steps of simulation and set it in timings
 *
 * @param lineElements
 *
 * @throws InputFileException if number of steps is negative
 */
void InputFileParserTimings::parseNumberOfSteps(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto numberOfSteps = stoi(lineElements[2]);

    if (numberOfSteps < 0) throw customException::InputFileException("Number of steps cannot be negative");

    _engine.getTimings().setNumberOfSteps(size_t(numberOfSteps));
}