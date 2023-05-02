#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

#include "rstFileReader.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace StringUtilities;
using namespace Setup::RstFileReader;

RstFileReader::RstFileReader(const string &filename, Settings &settings) : _filename(filename), _settings(settings) {
    _sections.push_back(new BoxSection);
    _sections.push_back(new NoseHooverSection);
    _sections.push_back(new StepCountSection);
}
RstFileReader::~RstFileReader() {
    for (RstFileSection *section : _sections)
        delete section;

    delete _atomSection;
}

RstFileSection *RstFileReader::determineSection(vector<string> &lineElements)
{
    for (RstFileSection *section : _sections)
    {
        if (section->keyword() == boost::algorithm::to_lower_copy(lineElements[0]))
            return section;
    }

    return _atomSection;
}

unique_ptr<SimulationBox> RstFileReader::read()
{
    auto simulationBox = make_unique<SimulationBox>(SimulationBox());
    string line;
    vector<string> lineElements;
    ifstream rstFile(_filename);
    int lineNumber = 1;

    if (rstFile.fail())
        throw runtime_error("\"" + _filename + "\"" + " File not found");

    while (getline(rstFile, line))
    {
        line = removeComments(line, "#");
        lineElements = splitString(line);

        auto section = determineSection(lineElements);
        section->_lineNumber = lineNumber++;
        section->process(lineElements, _settings, *simulationBox);
    }

    return simulationBox;
}

unique_ptr<SimulationBox> read_rst(string filename, Settings &settings)
{
    RstFileReader rstFileReader(filename, settings);
    return rstFileReader.read();
}