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

#include "restartFileReader.hpp"

#include <fstream>   // for basic_istream, ifstream
#include <string>    // for basic_string, string

#include "boxSection.hpp"          // for BoxSection
#include "engine.hpp"              // for Engine
#include "fileSettings.hpp"        // for FileSettings
#include "noseHooverSection.hpp"   // for NoseHooverSection
#include "stepCountSection.hpp"    // for StepCountSection
#include "stringUtilities.hpp"     // for removeComments, splitString

using namespace input::restartFile;
using namespace engine;
using namespace utilities;
using namespace settings;

/**
 * @brief Construct a new Rst File Reader:: Rst File Reader object
 *
 * @details The constructor initializes the sections of the .rst file and pushes
 * them into a vector. It also sets the filename and the engine object and with
 * the filename it opens the file in the ifstream object _fp.
 *
 *
 * @param filename
 * @param engine
 */
RestartFileReader::RestartFileReader(
    const std::string &filename,
    Engine            &engine
)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _sections.push_back(std::make_unique<BoxSection>());
    _sections.push_back(std::make_unique<NoseHooverSection>());
    _sections.push_back(std::make_unique<StepCountSection>());
}

/**
 * @brief Determines which section of the .rst file the line belongs to
 *
 * @param lineElements
 * @return RestartFileSection*
 */
RestartFileSection *RestartFileReader::determineSection(
    std::vector<std::string> &lineElements
)
{
    for (const auto &section : _sections)
        if (section->keyword() == toLowerCopy(lineElements[0]))
            return section.get();

    return _atomSection.get();
}

/**
 * @brief Reads a restart file and calls the process function of the
 * corresponding section
 *
 * @throw customException::InputFileException if file not found
 */
void RestartFileReader::read()
{
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

        auto *section        = determineSection(lineElements);
        section->_lineNumber = lineNumber;
        section->_fp         = &_fp;
        section->process(lineElements, _engine);
        lineNumber = section->_lineNumber;
        ++lineNumber;
    }
}

/**
 * @brief wrapper function to construct a RestartFileReader object and call the
 * read function
 *
 * @param engine
 */
void input::restartFile::readRestartFile(Engine &engine)
{
    const auto filename = FileSettings::getStartFileName();

    engine.getStdoutOutput().writeRead("Start File", filename);
    engine.getLogOutput().writeRead("Start File", filename);

    RestartFileReader rstFileReader(filename, engine);
    rstFileReader.read();
}