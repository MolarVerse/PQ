#include "exceptions.hpp"
#include "inputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace customException;

void InputFileReader::parseTopologyFilename(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    const auto filename = lineElements[2];

    if (filename.empty()) throw InputFileException("Topologyfilename cannot be empty");

    if (!fopen(filename.c_str(), "r")) throw InputFileException("Cannot open topology file - filename = " + string(filename));

    _engine.getSettings().setTopologyFilename(filename);
}