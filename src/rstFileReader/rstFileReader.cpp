#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "rstFileReader.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace StringUtilities;

RstFileReader::RstFileReader(string filename, Settings &settings) : _filename(filename), _settings(settings) {}
RstFileReader::~RstFileReader() {}

unique_ptr<SimulationBox> RstFileReader::read()
{
    auto simulationBox = unique_ptr<SimulationBox>(new SimulationBox);

    cout << "Reading file " << _filename << endl;

    ifstream rstFile(_filename);
    if (rstFile.fail())
        throw runtime_error("\"" + _filename + "\"" + " File not found");

    string line;

    cout << _settings.getStepCount() << endl;

    while (getline(rstFile, line))
    {
        line = removeComments(line, "#");
        // cout << line << endl;
    }

    _settings.setStepCount(-1);

    return simulationBox;
}

unique_ptr<SimulationBox> read_rst(string filename, Settings &settings)
{
    RstFileReader rstFileReader(filename, settings);
    return rstFileReader.read();
}