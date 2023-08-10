#include "inputFileParserResetKinetics.hpp"

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

/**
 * @brief parse nscale and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if nscale is negative
 */
void InputFileParserResetKinetics::parseNScale(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto nScale = stoi(lineElements[2]);
    if (nScale < 0)
        throw InputFileException("Nscale must be positive");
    else
        _engine.getSettings().setNScale(static_cast<size_t>(nScale));
}

/**
 * @brief parse fscale and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if fscale is negative
 */
void InputFileParserResetKinetics::parseFScale(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto fScale = stoi(lineElements[2]);
    if (fScale < 0)
        throw InputFileException("Fscale must be positive");
    else
        _engine.getSettings().setFScale(static_cast<size_t>(fScale));
}

/**
 * @brief parse nreset and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if nreset is negative
 */
void InputFileParserResetKinetics::parseNReset(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto nReset = stoi(lineElements[2]);
    if (nReset < 0)
        throw InputFileException("Nreset must be positive");
    else
        _engine.getSettings().setNReset(static_cast<size_t>(nReset));
}

/**
 * @brief parse freset and set it in settings
 *
 * @param lineElements
 *
 * @throw InputFileException if freset is negative
 */
void InputFileParserResetKinetics::parseFReset(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto fReset = stoi(lineElements[2]);
    if (fReset < 0)
        throw InputFileException("Freset must be positive");
    else
        _engine.getSettings().setFReset(static_cast<size_t>(fReset));
}
