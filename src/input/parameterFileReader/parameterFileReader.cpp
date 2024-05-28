/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "parameterFileReader.hpp"

#include "angleSection.hpp"              // for AngleSection
#include "bondSection.hpp"               // for BondSection
#include "dihedralSection.hpp"           // for DihedralSection
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for InputFileException, ParameterFileExce...
#include "fileSettings.hpp"              // for FileSettings
#include "forceFieldSettings.hpp"        // for ForceFieldSettings
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "nonCoulombicsSection.hpp"      // for NonCoulombicsSection
#include "stringUtilities.hpp"           // for removeComments, splitString, toLowerCopy
#include "typesSection.hpp"              // for TypesSection

#include <functional>   // for identity
#include <ranges>       // for __find_if_fn, find_if

using namespace input::parameterFile;

/**
 * @brief constructor
 *
 * @details initializes file pointer _fp with filename and adds all parameter file sections to _parameterFileSections
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(const std::string &filename, engine::Engine &engine)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _parameterFileSections.push_back(std::make_unique<TypesSection>());
    _parameterFileSections.push_back(std::make_unique<BondSection>());
    _parameterFileSections.push_back(std::make_unique<AngleSection>());
    _parameterFileSections.push_back(std::make_unique<DihedralSection>());
    _parameterFileSections.push_back(std::make_unique<ImproperDihedralSection>());
    _parameterFileSections.push_back(std::make_unique<NonCoulombicsSection>());
}

/**
 * @brief determines which section of the parameter file the header line belongs to
 *
 * @param lineElements
 * @return ParameterFileSection*
 *
 * @throws customException::ParameterFileException if unknown or already parsed keyword
 */
ParameterFileSection *ParameterFileReader::determineSection(const std::vector<std::string> &lineElements)
{
    const auto iterEnd = _parameterFileSections.end();

    for (auto section = _parameterFileSections.begin(); section != iterEnd; ++section)
        if ((*section)->keyword() == utilities::toLowerCopy(lineElements[0]))
            return (*section).get();

    throw customException::ParameterFileException("Unknown or already parsed keyword \"" + lineElements[0] +
                                                  "\" in parameter file");
}

/**
 * @brief deletes section from _parameterFileSections
 *
 * @param section
 */
void ParameterFileReader::deleteSection(const ParameterFileSection *section)
{
    auto       sectionIsEqual = [section](auto &sectionUniquePtr) { return sectionUniquePtr.get() == section; };
    const auto result         = std::ranges::find_if(_parameterFileSections, sectionIsEqual);
    _parameterFileSections.erase(result);
}

/**
 * @brief reads parameter file
 *
 * @details Reads parameter file and according to the first word of the line, the corresponding section is called and then the
 * line is processed. After processing the line, the section is deleted from _parameterFileSections.
 *
 * @throws customException::InputFileException if file was not provided
 * @throws customException::InputFileException if file does not exist
 */
void ParameterFileReader::read()
{
    if (!settings::FileSettings::isParameterFileNameSet())
        throw customException::InputFileException("Parameter file needed for requested simulation setup");

    std::string line;
    int         lineNumber = 1;

    while (getline(_fp, line))
    {
        line              = utilities::removeComments(line, "#");
        auto lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++lineNumber;
            continue;
        }

        auto *section = determineSection(lineElements);
        ++lineNumber;
        section->setLineNumber(lineNumber);
        section->setFp(&_fp);
        section->process(lineElements, _engine);
        lineNumber = section->getLineNumber();

        deleteSection(section);
    }
}

/**
 * @brief constructs a ParameterFileReader and reads parameter file
 *
 * @param engine
 */
void input::parameterFile::readParameterFile(engine::Engine &engine)
{
    if (!isNeeded())
        return;

    engine.getStdoutOutput().writeRead("Parameter File", settings::FileSettings::getParameterFilename());
    engine.getLogOutput().writeRead("Parameter File", settings::FileSettings::getParameterFilename());

    ParameterFileReader parameterFileReader(settings::FileSettings::getParameterFilename(), engine);
    parameterFileReader.read();
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if force field is activated
 * @return false
 */
bool input::parameterFile::isNeeded()
{
    if (settings::ForceFieldSettings::isActive())
        return true;

    return false;
}