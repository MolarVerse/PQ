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

RstFileReader::RstFileReader(const string &filename, Engine &engine) : _filename(filename), _fp(filename), _engine(engine)
{
    _sections.push_back(new BoxSection);
    _sections.push_back(new NoseHooverSection);
    _sections.push_back(new StepCountSection);
}
RstFileReader::~RstFileReader()
{
    for (const RstFileSection *section : _sections)
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
    int lineNumber = 1;

    if (_fp.fail())
        throw InputFileException("\"" + _filename + "\"" + " File not found");

    while (getline(_fp, line))
    {
        line = removeComments(line, "#");
        lineElements = splitString(line);

        if (lineElements.empty())
            continue;

        auto *section = determineSection(lineElements);
        section->_lineNumber = lineNumber++;
        section->_fp = &_fp;
        section->process(lineElements, _engine);
    }
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @param engine
 */
void readRstFile(Engine &engine)
{
    RstFileReader rstFileReader(engine.getSettings().getStartFilename(), engine);
    rstFileReader.read();
}