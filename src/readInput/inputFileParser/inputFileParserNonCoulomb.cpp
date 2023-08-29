#include "inputFileParserNonCoulomb.hpp"

#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) noncoulomb <string>
 *
 * @param engine
 */
InputFileParserNonCoulomb::InputFileParserNonCoulomb(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("noncoulomb"), bind_front(&InputFileParserNonCoulomb::parseNonCoulombType, this), false);
}

/**
 * @brief Parse the nonCoulombic type of the guff.dat file
 *
 * @details Possible options are:
 * 1) "guff"  - guff.dat file is used (default)
 * 2) "lj"
 * 3) "buck"
 * 4) "morse"
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if invalid nonCoulomb type
 */
void InputFileParserNonCoulomb::parseNonCoulombType(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    if (lineElements[2] == "guff")
        settings::PotentialSettings::setNonCoulombType("guff");
    else if (lineElements[2] == "lj")
        settings::PotentialSettings::setNonCoulombType("lj");
    else if (lineElements[2] == "buck")
        settings::PotentialSettings::setNonCoulombType("buck");
    else if (lineElements[2] == "morse")
        settings::PotentialSettings::setNonCoulombType("morse");
    else
        throw customException::InputFileException(
            format("Invalid nonCoulomb type \"{}\" at line {} in input file. Possible options are: lj, buck, morse and guff",
                   lineElements[2],
                   lineNumber));
}