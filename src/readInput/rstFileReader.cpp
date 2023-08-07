#include "rstFileReader.hpp"

#include "exceptions.hpp"
#include "stringUtilities.hpp"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace utilities;
using namespace readInput;
using namespace engine;
using namespace customException;

RstFileReader::RstFileReader(const string &filename, Engine &engine) : _filename(filename), _fp(filename), _engine(engine)
{
    _sections.push_back(make_unique<BoxSection>());
    _sections.push_back(make_unique<NoseHooverSection>());
    _sections.push_back(make_unique<StepCountSection>());
}

/**
 * @brief Determines which section of the .rst file the line belongs to
 *
 * @param lineElements
 * @return RstFileSection*
 */
RstFileSection *RstFileReader::determineSection(vector<string> &lineElements)
{
    for (const auto &section : _sections)
        if (section->keyword() == boost::algorithm::to_lower_copy(lineElements[0])) return section.get();

    return _atomSection.get();
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @throw InputFileException if file not found
 */
void RstFileReader::read()
{
    string         line;
    vector<string> lineElements;
    int            lineNumber = 1;

    if (_fp.fail()) throw InputFileException("\"" + _filename + "\"" + " File not found");

    while (getline(_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        if (lineElements.empty())
        {
            ++lineNumber;
            continue;
        }

        auto *section        = determineSection(lineElements);
        section->_lineNumber = lineNumber++;
        section->_fp         = &_fp;
        section->process(lineElements, _engine);
        lineNumber = section->_lineNumber;
    }
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @param engine
 */
void readInput::readRstFile(Engine &engine)
{
    RstFileReader rstFileReader(engine.getSettings().getStartFilename(), engine);
    rstFileReader.read();
}