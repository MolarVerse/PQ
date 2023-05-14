#include "guffDatReader.hpp"
#include "stringUtilities.hpp"
#include "exceptions.hpp"

#include <fstream>

using namespace std;
using namespace StringUtilities;

void readGuffDat(Engine &engine)
{
    GuffDatReader guffDat(engine);
    guffDat.read();
}

void GuffDatReader::read()
{
    ifstream fp(_filename);
    string line;

    while (getline(fp, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            _lineNumber++;
            continue;
        }

        auto lineCommands = getLineCommands(line, _lineNumber);

        if (lineCommands.size() != 23)
            throw GuffDatException("Invalid number of commands in line " + to_string(_lineNumber));

        parseLine(lineCommands);

        _lineNumber++;
    }
}

void GuffDatReader::parseLine(vector<string> &lineCommands)
{
    try
    {
        auto molecule1 = _engine.getSimulationBox().findMoleculeType(stoi(lineCommands[0]));
    }
    catch (const UserInputException &e)
    {
        cout << e.what() << endl
             << endl;
    }
}