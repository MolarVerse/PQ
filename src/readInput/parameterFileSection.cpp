#include "parameterFileSection.hpp"

#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput::parameterFile;

/**
 * @brief general process function for parameter sections
 *
 * @param line
 * @param engine
 */
void ParameterFileSection::process(vector<string> &lineElements, engine::Engine &engine)
{
    string line;
    auto   endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = StringUtilities::removeComments(line, "#");
        lineElements = StringUtilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (StringUtilities::toLowerCopy(lineElements[0]) == "end")
        {
            ++_lineNumber;
            endedNormal = true;
            break;
        }

        processSection(lineElements, engine);

        ++_lineNumber;
    }

    endedNormally(endedNormal);
}

/**
 * @brief dummy function to forward process of one line to processSection
 *
 * @param lineElements
 * @param engine
 */
void TypesSection::process(vector<string> &lineElements, engine::Engine &engine) { processSection(lineElements, engine); }

/**
 * @brief process types section
 *
 * @param lineElements
 * @param engine
 */
void TypesSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 8)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 8!");

    const auto scaleCoulomb     = stod(lineElements[6]);
    const auto scaleVanDerWaals = stod(lineElements[7]);

    if (scaleCoulomb < 0.0 || scaleCoulomb > 1.0)
        throw customException::ParameterFileException("Wrong scaleCoulomb in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - has to be between 0 and 1!");

    if (scaleVanDerWaals < 0.0 || scaleVanDerWaals > 1.0)
        throw customException::ParameterFileException("Wrong scaleVanDerWaals in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - has to be between 0 and 1!");

    engine.getForceField().setScale14Coulomb(scaleCoulomb);
    engine.getForceField().setScale14VanDerWaals(scaleVanDerWaals);
}

/**
 * @brief check if section ended normally
 *
 * @param endedNormally
 */
void ParameterFileSection::endedNormally(bool endedNormally)
{
    if (!endedNormally)
        throw customException::ParameterFileException("Parameter file " + keyword() + " section ended abnormally!");
}