#include "rstFileReader.hpp"

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException
#include "settings.hpp"          // for Settings
#include "stringUtilities.hpp"   // for removeComments, splitString

#include <boost/algorithm/string/case_conv.hpp>   // for to_lower_copy
#include <boost/iterator/iterator_facade.hpp>     // for operator!=
#include <format>                                 // for format
#include <fstream>                                // for basic_istream, ifstream
#include <string>                                 // for basic_string, string

using namespace std;
using namespace utilities;
using namespace readInput;
using namespace engine;
using namespace customException;

/**
 * @brief Construct a new Rst File Reader:: Rst File Reader object
 *
 * @param filename
 * @param engine
 */
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
        if (section->keyword() == boost::algorithm::to_lower_copy(lineElements[0]))
            return section.get();

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

    if (_fp.fail())
        throw InputFileException(format(R"("{}" File not found)", _filename));

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