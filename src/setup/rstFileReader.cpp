#include <string>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

#include "rstFileReader.hpp"
#include "stringUtilities.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace StringUtilities;
using namespace Setup::RstFileReader;

RstFileReader::RstFileReader(const string &filename, Engine &engine) : _filename(filename), _engine(engine)
{
    _sections.push_back(new BoxSection);
    _sections.push_back(new NoseHooverSection);
    _sections.push_back(new StepCountSection);
}
RstFileReader::~RstFileReader()
{
    for (RstFileSection *section : _sections)
        delete section;

    delete _atomSection;
}

/**
 * @brief Determines which section of the .rst file the line belongs to
 *
 * @param lineElements
 * @return RstFileSection*
 */
RstFileSection *RstFileReader::determineSection(vector<string> &lineElements)
{
    for (RstFileSection *section : _sections)
    {
        if (section->keyword() == boost::algorithm::to_lower_copy(lineElements[0]))
            return section;
    }

    return _atomSection;
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @throw InputFileException if file not found
 */
void RstFileReader::read()
{
    string line;
    vector<string> lineElements;
    ifstream rstFile(_filename);
    int lineNumber = 1;

    if (rstFile.fail())
        throw InputFileException("\"" + _filename + "\"" + " File not found");

    while (getline(rstFile, line))
    {
        line = removeComments(line, "#");
        lineElements = splitString(line);

        auto section = determineSection(lineElements);
        section->_lineNumber = lineNumber++;
        section->process(lineElements, _engine);
    }
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @param engine
 */
void read_rst(Engine &engine)
{
    RstFileReader rstFileReader(engine._settings.getStartFilename(), engine);
}