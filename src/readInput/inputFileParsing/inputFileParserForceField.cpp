#include "inputFileParserForceField.hpp"

#include "forceFieldNonCoulomb.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Force Field:: Input File Parser Force Field object
 *
 * @param engine
 */
InputFileParserForceField::InputFileParserForceField(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("force-field"), bind_front(&InputFileParserForceField::parseForceFieldType, this), false);
}

/**
 * @brief Parse the integrator used in the simulation
 *
 * @param lineElements
 *
 * @throws InputFileException if force-field is not valid - currently only on, off and bonded are supported
 */
void InputFileParserForceField::parseForceFieldType(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
    {
        _engine.getForceFieldPtr()->activate();
        _engine.getForceFieldPtr()->activateNonCoulombic();
        _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());   // TODO: test this one
    }
    else if (lineElements[2] == "off")
    {
        _engine.getForceFieldPtr()->deactivate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else if (lineElements[2] == "bonded")
    {
        _engine.getForceFieldPtr()->activate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else
        throw InputFileException(
            format(R"(Invalid force-field keyword "{}" at line {} in input file - possible keywords are "on", "off" or "bonded")",
                   lineElements[2],
                   lineNumber));
}