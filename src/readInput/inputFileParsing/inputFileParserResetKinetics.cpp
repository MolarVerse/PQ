#include "inputFileParser.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Reset Kinetics:: Input File Parser Reset Kinetics object
 *
 * @param engine
 */
InputFileParserResetKinetics::InputFileParserResetKinetics(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("nscale"), bind_front(&InputFileParserResetKinetics::parseNScale, this), false);
    addKeyword(string("fscale"), bind_front(&InputFileParserResetKinetics::parseFScale, this), false);
    addKeyword(string("nreset"), bind_front(&InputFileParserResetKinetics::parseNReset, this), false);
    addKeyword(string("freset"), bind_front(&InputFileParserResetKinetics::parseFReset, this), false);
}

void InputFileParserResetKinetics::parseNScale(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto nScale = stoi(lineElements[2]);
    if (nScale < 0)
        throw InputFileException("Invalid nscale \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file");
    else
        _engine.getSettings().setNScale(static_cast<size_t>(nScale));
}

void InputFileParserResetKinetics::parseFScale(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto fScale = stoi(lineElements[2]);
    if (fScale < 0)
        throw InputFileException("Invalid fscale \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file");
    else
        _engine.getSettings().setFScale(static_cast<size_t>(fScale));
}

void InputFileParserResetKinetics::parseNReset(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto nReset = stoi(lineElements[2]);
    if (nReset < 0)
        throw InputFileException("Invalid nreset \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file");
    else
        _engine.getSettings().setNReset(static_cast<size_t>(nReset));
}

void InputFileParserResetKinetics::parseFReset(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto fReset = stoi(lineElements[2]);
    if (fReset < 0)
        throw InputFileException("Invalid freset \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file");
    else
        _engine.getSettings().setFReset(static_cast<size_t>(fReset));
}
