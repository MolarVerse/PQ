#include "inputFileParserTopology.hpp"

#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput;
using namespace customException;
using namespace utilities;

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
 *
 * @throws InputFileException if topology filename is empty or file does not exist
 */
void InputFileParserTopology::parseTopologyFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (filename.empty()) throw InputFileException("Topology filename cannot be empty");

    if (!fileExists(filename)) throw InputFileException("Cannot open topology file - filename = " + filename);

    _engine.getSettings().setTopologyFilename(filename);
}