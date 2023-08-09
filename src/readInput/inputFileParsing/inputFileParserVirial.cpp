#include "inputFileParser.hpp"

#include <iostream>

using namespace std;
using namespace readInput;
using namespace virial;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Virial:: Input File Parser Virial object
 *
 * @param engine
 */
InputFileParserVirial::InputFileParserVirial(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("virial"), bind_front(&InputFileParserVirial::parseVirial, this), false);
}

/**
 * @brief parses virial command
 *
 * @param lineElements
 */
void InputFileParserVirial::parseVirial(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "molecular")
    {
        _engine.makeVirial(VirialMolecular());
    }
    else if (lineElements[2] == "atomic")
    {
        _engine.makeVirial(VirialAtomic());
        _engine.getPhysicalData().changeKineticVirialToAtomic();
    }
    else
        throw InputFileException("Invalid virial setting \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                                 "in input file");
}