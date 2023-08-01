#include "topologyReader.hpp"

#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput;
using namespace StringUtilities;

/**
 * @brief constructor
 *
 * @param filename
 * @param engine
 */
TopologyReader::TopologyReader(const string &filename, engine::Engine &engine)
    : _filename(filename), _fp(filename), _engine(engine)
{
    _topologySections.push_back(new ShakeSection());
    _topologySections.push_back(new BondSection());
    _topologySections.push_back(new AngleSection());
    _topologySections.push_back(new DihedralSection());
    _topologySections.push_back(new ImproperDihedralSection());
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if shake is activated
 * @return false
 */
bool TopologyReader::isNeeded() const
{
    if (_engine.getConstraints().isActivated()) return true;

    if (_engine.getForceField().isActivated()) return true;

    return false;
}

/**
 * @brief reads topology file
 */
void TopologyReader::read()
{
    string         line;
    vector<string> lineElements;
    int            lineNumber = 1;

    if (!isNeeded()) return;

    if (_filename.empty()) throw customException::InputFileException("Topology file needed for requested simulation setup");

    if (!filesystem::exists(_filename))
        throw customException::InputFileException("Topology file \"" + _filename + "\"" + " File not found");

    while (getline(_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

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
 */
TopologySection *TopologyReader::determineSection(const vector<string> &lineElements)
{
    const auto iterEnd = _topologySections.end();

    for (auto section = _topologySections.begin(); section != iterEnd; ++section)
        if ((*section)->keyword() == to_lower_copy(lineElements[0])) return *section;

    throw customException::TopologyException("Unknown or already passed keyword \"" + lineElements[0] + "\" in topology file");
}

/**
 * @brief constructs a TopologyReader and reads topology file
 *
 * @param filename
 * @param engine
 */
void readInput::readTopologyFile(engine::Engine &engine)
{
    TopologyReader topologyReader(engine.getSettings().getTopologyFilename(), engine);
    topologyReader.read();
}