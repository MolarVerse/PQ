#include "topologyReader.hpp"

#include "angleSection.hpp"              // for AngleSection
#include "bondSection.hpp"               // for BondSection
#include "constraints.hpp"               // for Constraints
#include "dihedralSection.hpp"           // for DihedralSection
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for InputFileException, TopologyException
#include "forceField.hpp"                // for ForceField
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "settings.hpp"                  // for Settings
#include "shakeSection.hpp"              // for ShakeSection
#include "stringUtilities.hpp"           // for removeComments, splitString, toLowerCopy

#include <filesystem>   // for exists
#include <string>       // for string, basic_string, operator==, operator!=
#include <vector>       // for vector

using namespace readInput::topology;

/**
 * @brief constructor
 *
 * @details Sets filename and engine - also initializes the file pointer _fp. Then all possible topology sections are added to
 * _topologySections.
 *
 * @param filename
 * @param engine
 */
TopologyReader::TopologyReader(const std::string &filename, engine::Engine &engine)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _topologySections.push_back(std::make_unique<ShakeSection>());
    _topologySections.push_back(std::make_unique<BondSection>());
    _topologySections.push_back(std::make_unique<AngleSection>());
    _topologySections.push_back(std::make_unique<DihedralSection>());
    _topologySections.push_back(std::make_unique<ImproperDihedralSection>());
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if shake is activated
 * @return true if force field is activated
 * @return false
 */
bool TopologyReader::isNeeded() const
{
    if (_engine.getConstraints().isActivated())
        return true;

    if (_engine.getForceField().isActivated())
        return true;

    return false;
}

/**
 * @brief reads topology file
 *
 * @details reads topology file line by line and determines which section the line belongs to. Then the line is processed by the
 * section.
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

    if (!isNeeded())
        return;

    if (_fileName.empty())
        throw customException::InputFileException("Topology file needed for requested simulation setup");

    if (!std::filesystem::exists(_fileName))
        throw customException::InputFileException("Topology file \"" + _fileName + "\"" + " File not found");

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
 * @brief determines which section of the topology file the header line belongs to
 *
 * @param lineElements
 * @return TopologySection*
 *
 * @throws customException::TopologyException if keyword is unknown or already parsed
 */
TopologySection *TopologyReader::determineSection(const std::vector<std::string> &lineElements)
{
    const auto iterEnd = _topologySections.end();

    for (auto section = _topologySections.begin(); section != iterEnd; ++section)
        if ((*section)->keyword() == utilities::toLowerCopy(lineElements[0]))
            return (*section).get();

    throw customException::TopologyException("Unknown or already parsed keyword \"" + lineElements[0] + "\" in topology file");
}

/**
 * @brief wrapper to construct a TopologyReader and reads topology file
 *
 * @param filename
 * @param engine
 */
void readInput::topology::readTopologyFile(engine::Engine &engine)
{
    TopologyReader topologyReader(engine.getSettings().getTopologyFilename(), engine);
    topologyReader.read();
}