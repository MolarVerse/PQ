#include "exceptions.hpp"
#include "inputFileParser.hpp"

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Topology:: Input File Parser Topology object
 *
 * @param engine
 */
InputFileParserTopology::InputFileParserTopology(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("topology_file"), bind_front(&InputFileParserTopology::parseTopologyFilename, this), false);
}

/**
 * @brief parse topology file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserTopology::parseTopologyFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (filename.empty()) throw InputFileException("Topology filename cannot be empty");

    if (!fopen(filename.c_str(), "r")) throw InputFileException("Cannot open topology file - filename = " + string(filename));

    _engine.getSettings().setTopologyFilename(filename);
}