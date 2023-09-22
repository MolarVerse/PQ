#include "inputFileParserForceField.hpp"

#include "engine.hpp"                 // for Engine
#include "exceptions.hpp"             // for InputFileException, customException
#include "forceFieldClass.hpp"        // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "forceFieldSettings.hpp"     // for ForceFieldSettings
#include "potential.hpp"              // for Potential
#include "stringUtilities.hpp"        // for toLowerCopy

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Force Field:: Input File Parser Force Field object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) force-field <on/off/bonded>
 *
 * @param engine
 */
InputFileParserForceField::InputFileParserForceField(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("force-field"), bind_front(&InputFileParserForceField::parseForceFieldType, this), false);
}

/**
 * @brief Parse the integrator used in the simulation
 *
 * @details Possible options are:
 * 1) "on"  - force-field is activated
 * 2) "off" - force-field is deactivated (default)
 * 3) "bonded" - only bonded interactions are activated
 *
 * @param lineElements
 *
 * @throws InputFileException if force-field is not valid - currently only on, off and bonded are supported
 */
void InputFileParserForceField::parseForceFieldType(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto forceFieldType = utilities::toLowerCopy(lineElements[2]);

    if (forceFieldType == "on")
    {
        settings::ForceFieldSettings::activate();
        _engine.getForceFieldPtr()->activateNonCoulombic();
        _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    }
    else if (forceFieldType == "off")
    {
        settings::ForceFieldSettings::deactivate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else if (forceFieldType == "bonded")
    {
        settings::ForceFieldSettings::activate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else
        throw customException::InputFileException(
            format(R"(Invalid force-field keyword "{}" at line {} in input file - possible keywords are "on", "off" or "bonded")",
                   forceFieldType,
                   lineNumber));
}