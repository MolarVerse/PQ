#include "parameterFileSection.hpp"

#include "angleType.hpp"              // for AngleType
#include "bondType.hpp"               // for BondType
#include "buckinghamPair.hpp"         // for BuckinghamPair
#include "constants.hpp"              // for _DEG_TO_RAD_
#include "coulombPotential.hpp"       // for CoulombPotential
#include "dihedralForceField.hpp"     // for DihedralForceField
#include "dihedralType.hpp"           // for DihedralType
#include "engine.hpp"                 // for Engine
#include "exceptions.hpp"             // for ParameterFileException
#include "forceField.hpp"             // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "intraNonBondedMap.hpp"      // for IntraNonBondedMap
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "morsePair.hpp"              // for MorsePair
#include "nonCoulombPotential.hpp"    // for NonCoulombPotential, NonCoulombType
#include "potential.hpp"              // for Potential
#include "stringUtilities.hpp"        // for toLowerCopy, removeComments, splitString

#include <format>    // for format
#include <istream>   // for basic_istream, ifstream, std
#include <memory>    // for allocator, make_shared

using namespace readInput::parameterFile;

/**
 * @brief reads a general parameter file section
 *
 * @details Calls processHeader at the beginning of each section and processSection for each line in the section.
 * If the "end" keyword is found, the section is ended normally.
 *
 * @param line
 * @param engine
 */
void ParameterFileSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    processHeader(lineElements, engine);

    std::string line;
    auto        endedNormal = false;

    while (getline(*_fp, line))
    {

        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
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
 * @brief check if section ended normally
 *
 * @param endedNormally
 *
 * @throw customException::ParameterFileException if section did not end normally
 */
void ParameterFileSection::endedNormally(bool endedNormally)
{
    if (!endedNormally)
        throw customException::ParameterFileException("Parameter file " + keyword() + " section ended abnormally!");
}