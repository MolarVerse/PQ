#include "inputFileParserVirial.hpp"

#include "engine.hpp"         // for Engine
#include "exceptions.hpp"     // for InputFileException, customException
#include "physicalData.hpp"   // for PhysicalData
#include "virial.hpp"         // for VirialAtomic, VirialMolecular, virial

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

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
    addKeyword(std::string("virial"), bind_front(&InputFileParserVirial::parseVirial, this), false);
}

/**
 * @brief parses virial command
 *
 * @param lineElements
 *
 * @throws InputFileException if invalid virial keyword
 */
void InputFileParserVirial::parseVirial(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
        throw InputFileException(format("Invalid virial setting \"{}\" at line {} in input file", lineElements[2], lineNumber));
}