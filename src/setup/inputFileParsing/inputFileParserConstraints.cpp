#include "exceptions.hpp"
#include "inputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace customException;

void InputFileReader::parseShakeActivated(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "on")
        _engine.getConstraints().activate();
    else if (lineElements[2] == "off")
    {
        // default
    }
    else
    {
        auto message  = "Invalid shake keyword \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file\n";
        message      += R"(Possible keywords are "on" and "off")";
        throw InputFileException(message);
    }
}