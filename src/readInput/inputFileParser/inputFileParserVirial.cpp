#include "inputFileParserVirial.hpp"

#include "engine.hpp"         // for Engine
#include "exceptions.hpp"     // for InputFileException, customException
#include "physicalData.hpp"   // for PhysicalData
#include "virial.hpp"         // for VirialAtomic, VirialMolecular, virial

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Virial:: Input File Parser Virial object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) virial <molecular/atomic>
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
 * @details possible options are:
 * 1) molecular - molecular virial (default)
 * 2) atomic    - atomic virial - sets file pointer to atomic virial file in physical data
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if invalid virial keyword
 */
void InputFileParserVirial::parseVirial(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "molecular")
    {
        _engine.makeVirial(virial::VirialMolecular());
    }
    else if (lineElements[2] == "atomic")
    {
        _engine.makeVirial(virial::VirialAtomic());
        _engine.getPhysicalData().changeKineticVirialToAtomic();
    }
    else
        throw customException::InputFileException(
            format("Invalid virial setting \"{}\" at line {} in input file", lineElements[2], lineNumber));
}