#include "parameterFileSection.hpp"

#include "exceptions.hpp"        // for ParameterFileException
#include "stringUtilities.hpp"   // for removeComments, splitString, toLowerCopy

#include <fstream>   // for getline

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