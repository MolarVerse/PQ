#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace setup;
using namespace customException;

void InputFileReader::parseNScale(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    const auto nScale = stoi(lineElements[2]);
    if (nScale < 0)
        throw InputFileException("Invalid nscale \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
    else
        _engine.getSettings().setNScale(static_cast<size_t>(nScale));
}

void InputFileReader::parseFScale(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    const auto fScale = stoi(lineElements[2]);
    if (fScale < 0)
        throw InputFileException("Invalid fscale \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
    else
        _engine.getSettings().setFScale(static_cast<size_t>(fScale));
}

void InputFileReader::parseNReset(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    const auto nReset = stoi(lineElements[2]);
    if (nReset < 0)
        throw InputFileException("Invalid nreset \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
    else
        _engine.getSettings().setNReset(static_cast<size_t>(nReset));
}

void InputFileReader::parseFReset(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    const auto fReset = stoi(lineElements[2]);
    if (fReset < 0)
        throw InputFileException("Invalid freset \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
    else
        _engine.getSettings().setFReset(static_cast<size_t>(fReset));
}
