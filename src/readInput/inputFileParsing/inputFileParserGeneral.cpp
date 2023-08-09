#include "exceptions.hpp"
#include "inputFileParser.hpp"

#include <iostream>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General object
 *
 * @param engine
 */
InputFileParserGeneral::InputFileParserGeneral(engine::Engine &engine) : InputFileParser(engine)
{

    addKeyword(string("jobtype"), bind_front(&InputFileParserGeneral::parseJobType, this), true);
    addKeyword(string("start_file"), bind_front(&InputFileParserGeneral::parseStartFilename, this), true);
    addKeyword(string("moldescriptor_file"), bind_front(&InputFileParserGeneral::parseMoldescriptorFilename, this), false);
    addKeyword(string("guff_path"), bind_front(&InputFileParserGeneral::parseGuffPath, this), false);
    addKeyword(string("guff_file"), bind_front(&InputFileParserGeneral::parseGuffDatFilename, this), false);
}

void InputFileParserGeneral::parseStartFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setStartFilename(lineElements[2]);
}

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileParserGeneral::parseMoldescriptorFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setMoldescriptorFilename(lineElements[2]);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileParserGeneral::parseGuffPath(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setGuffPath(lineElements[2]);
}

/**
 * @brief parse guff dat file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileParserGeneral::parseGuffDatFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setGuffDatFilename(lineElements[2]);
}

/**
 * @brief parse jobtype of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if jobtype is not recognised
 */
void InputFileParserGeneral::parseJobType(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "mm-md")
        _engine.getSettings().setJobtype("MMMD");
    else
        throw InputFileException("Invalid jobtype \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                                 "in input file");
}