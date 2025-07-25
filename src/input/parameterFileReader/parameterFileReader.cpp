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

#include <functional>   // for identity
#include <ranges>       // for __find_if_fn, find_if

#include "angleSection.hpp"      // for AngleSection
#include "bondSection.hpp"       // for BondSection
#include "dihedralSection.hpp"   // for DihedralSection
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException, ParameterFileExce...
#include "fileSettings.hpp"      // for FileSettings
#include "forceFieldSettings.hpp"        // for ForceFieldSettings
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "jCouplingSection.hpp"          // for JCouplingSection
#include "nonCoulombicsSection.hpp"      // for NonCoulombicsSection
#include "stringUtilities.hpp"   // for removeComments, splitString, toLowerCopy
#include "typesSection.hpp"      // for TypesSection

using namespace input::parameterFile;
using namespace engine;
using namespace utilities;
using namespace customException;
using namespace settings;

using std::make_unique;
using std::ranges::find_if;

/**
 * @brief constructor
 *
 * @details initializes file pointer _fp with filename and adds all parameter
 * file sections to _parameterFileSections
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(
    const std::string &filename,
    Engine            &engine
)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _parameterFileSections.push_back(make_unique<TypesSection>());
    _parameterFileSections.push_back(make_unique<BondSection>());
    _parameterFileSections.push_back(make_unique<AngleSection>());
    _parameterFileSections.push_back(make_unique<DihedralSection>());
    _parameterFileSections.push_back(make_unique<ImproperDihedralSection>());
    _parameterFileSections.push_back(make_unique<JCouplingSection>());
    _parameterFileSections.push_back(make_unique<NonCoulombicsSection>());
}

/**
 * @brief determines which section of the parameter file the header line belongs
 * to
 *
 * @param lineElements
 * @return ParameterFileSection*
 *
 * @throws ParameterFileException if unknown or already parsed
 * keyword
 */
ParameterFileSection *ParameterFileReader::determineSection(
    const std::vector<std::string> &lineElements
)
{
    const auto iterStart = _parameterFileSections.begin();
    const auto iterEnd   = _parameterFileSections.end();

    for (auto section = iterStart; section != iterEnd; ++section)
        if ((*section)->keyword() ==
            toLowerAndReplaceDashesCopy(lineElements[0]))
            return (*section).get();

    throw ParameterFileException(
        "Unknown or already parsed keyword \"" + lineElements[0] +
        "\" in parameter file"
    );
}

/**
 * @brief deletes section from _parameterFileSections
 *
 * @param section
 */
void ParameterFileReader::deleteSection(const ParameterFileSection *section)
{
    auto sectionIsEqual = [section](auto &sectionUniquePtr)
    { return sectionUniquePtr.get() == section; };

    const auto result = find_if(_parameterFileSections, sectionIsEqual);
    _parameterFileSections.erase(result);
}

/**
 * @brief reads parameter file
 *
 * @details Reads parameter file and according to the first word of the line,
 * the corresponding section is called and then the line is processed. After
 * processing the line, the section is deleted from _parameterFileSections.
 *
 * @throws InputFileException if file was not provided
 * @throws InputFileException if file does not exist
 */
void ParameterFileReader::read()
{
    if (!FileSettings::isParameterFileNameSet())
        throw InputFileException(
            "Parameter file needed for requested simulation setup"
        );

    std::string line;
    int         lineNumber = 1;

    while (getline(_fp, line))
    {
        line              = removeComments(line, "#");
        auto lineElements = splitString(line);

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
void input::parameterFile::readParameterFile(Engine &engine)
{
    if (!isNeeded())
        return;

    const auto filename = FileSettings::getParameterFilename();

    engine.getStdoutOutput().writeRead("Parameter File", filename);
    engine.getLogOutput().writeRead("Parameter File", filename);

    ParameterFileReader parameterFileReader(filename, engine);
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
    if (ForceFieldSettings::isActive())
        return true;

    return false;
}

/**************************************
 *                                    *
 * standard getter and setter methods *
 *                                    *
 **************************************/

/**
 * @brief set filename of parameter file
 *
 * @param filename
 */
void ParameterFileReader::setFilename(const std::string_view &filename)
{
    _fileName = filename;
}

/**
 * @brief get parameter file sections
 *
 * @return std::vector<std::unique_ptr<ParameterFileSection>>&
 */
std::vector<std::unique_ptr<ParameterFileSection>> &ParameterFileReader::
    getParameterFileSections()
{
    return _parameterFileSections;
}

/**
 * @brief get filename of parameter file
 *
 * @return const std::string&
 */
const std::string &ParameterFileReader::getFilename() const
{
    return _fileName;
}