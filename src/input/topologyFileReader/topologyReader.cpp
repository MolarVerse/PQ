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

#include "topologyReader.hpp"

#include <string>   // for string, basic_string, operator==, operator!=
#include <vector>   // for vector

#include "angleSection.hpp"                 // for AngleSection
#include "bondSection.hpp"                  // for BondSection
#include "constraints.hpp"                  // for Constraints
#include "dihedralSection.hpp"              // for DihedralSection
#include "distanceConstraintsSection.hpp"   // for DistanceConstraintsSection
#include "engine.hpp"                       // for Engine
#include "exceptions.hpp"           // for InputFileException, TopologyException
#include "fileSettings.hpp"         // for FileSettings
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "jCouplingSection.hpp"          // for JCouplingSection
#include "shakeSection.hpp"              // for ShakeSection
#include "stringUtilities.hpp"   // for removeComments, splitString, toLowerCopy

using namespace input::topology;

/**
 * @brief constructor
 *
 * @details Sets filename and engine - also initializes the file pointer _fp.
 * Then all possible topology sections are added to _topologySections.
 *
 * @param filename
 * @param engine
 */
TopologyReader::TopologyReader(
    const std::string &filename,
    engine::Engine    &engine
)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _topologySections.push_back(std::make_unique<ShakeSection>());
    _topologySections.push_back(std::make_unique<BondSection>());
    _topologySections.push_back(std::make_unique<AngleSection>());
    _topologySections.push_back(std::make_unique<DihedralSection>());
    _topologySections.push_back(std::make_unique<ImproperDihedralSection>());
    _topologySections.push_back(std::make_unique<DistanceConstraintsSection>());
    _topologySections.push_back(std::make_unique<JCouplingSection>());
}

/**
 * @brief reads topology file
 *
 * @details reads topology file line by line and determines which section the
 * line belongs to. Then the line is processed by the section.
 *
 * @throws customException::InputFileException if topology file is not set
 * @throws customException::InputFileException if topology file does not exist
 *
 */
void TopologyReader::read()
{
    std::string              line;
    std::vector<std::string> lineElements;
    int                      lineNumber = 1;

    if (!settings::FileSettings::isTopologyFileNameSet())
        throw customException::InputFileException(
            "Topology file needed for requested simulation setup"
        );

    while (getline(_fp, line))
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

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
    }
}

/**
 * @brief determines which section of the topology file the header line belongs
 * to
 *
 * @param lineElements
 * @return TopologySection*
 *
 * @throws customException::TopologyException if keyword is unknown or already
 * parsed
 */
TopologySection *TopologyReader::determineSection(
    const std::vector<std::string> &lineElements
)
{
    const auto iterStart = _topologySections.begin();
    const auto iterEnd   = _topologySections.end();

    for (auto section = iterStart; section != iterEnd; ++section)
        if ((*section)->keyword() == utilities::toLowerCopy(lineElements[0]))
            return (*section).get();

    throw customException::TopologyException(
        "Unknown or already parsed keyword \"" + lineElements[0] +
        "\" in topology file"
    );
}

/**
 * @brief wrapper to construct a TopologyReader and reads topology file
 *
 * @param filename
 * @param engine
 */
void input::topology::readTopologyFile(engine::Engine &engine)
{
    if (!isNeeded(engine))
        return;

    engine.getStdoutOutput().writeRead(
        "Topology File",
        settings::FileSettings::getTopologyFileName()
    );
    engine.getLogOutput().writeRead(
        "Topology File",
        settings::FileSettings::getTopologyFileName()
    );

    TopologyReader topologyReader(
        settings::FileSettings::getTopologyFileName(),
        engine
    );
    topologyReader.read();
}

/**
 * @brief checks if reading topology file is needed
 *
 * @param engine
 *
 * @return true if shake is activated
 * @return true if force field is activated
 * @return false
 */
bool input::topology::isNeeded(engine::Engine &engine)
{
    if (engine.getConstraints().isActive())
        return true;

    if (settings::ForceFieldSettings::isActive())
        return true;

    return false;
}