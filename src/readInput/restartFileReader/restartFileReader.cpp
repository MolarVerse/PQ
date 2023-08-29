#include "restartFileReader.hpp"

#include "boxSection.hpp"          // for BoxSection
#include "exceptions.hpp"          // for customException::InputFileException
#include "fileSettings.hpp"        // for FileSettings
#include "noseHooverSection.hpp"   // for NoseHooverSection
#include "stepCountSection.hpp"    // for StepCountSection
#include "stringUtilities.hpp"     // for removeComments, splitString

#include <format>    // for format
#include <fstream>   // for basic_istream, ifstream
#include <string>    // for basic_string, string

using namespace readInput::restartFile;

/**
 * @brief Construct a new Rst File Reader:: Rst File Reader object
 *
 * @details The constructor initializes the sections of the .rst file and pushes them into a vector. It also sets the filename and
 * the engine object and with the filename it opens the file in the ifstream object _fp.
 *
 *
 * @param filename
 * @param engine
 */
RestartFileReader::RestartFileReader(const std::string &filename, engine::Engine &engine)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _sections.push_back(std::make_unique<BoxSection>());
    _sections.push_back(std::make_unique<NoseHooverSection>());
    _sections.push_back(std::make_unique<StepCountSection>());
}

/**
 * @brief Determines which section of the .rst file the line belongs to
 *
 * @param lineElements
 * @return RestartFileSection*
 */
RestartFileSection *RestartFileReader::determineSection(std::vector<std::string> &lineElements)
{
    for (const auto &section : _sections)
        if (section->keyword() == utilities::toLowerCopy(lineElements[0]))
            return section.get();

    return _atomSection.get();
}

/**
 * @brief Reads a restart file and calls the process function of the corresponding section
 *
 * @throw customException::InputFileException if file not found
 */
void RestartFileReader::read()
{
    if (_fp.fail())
        throw customException::InputFileException(std::format(R"("{}" File not found)", _fileName));

    std::string line;
    int         lineNumber = 1;

    while (getline(_fp, line))
    {
        line              = utilities::removeComments(line, "#");
        auto lineElements = utilities::splitString(line);

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
 * @brief wrapper function to construct a RestartFileReader object and call the read function
 *
 * @param engine
 */
void readInput::restartFile::readRestartFile(engine::Engine &engine)
{
    RestartFileReader rstFileReader(settings::FileSettings::getStartFileName(), engine);
    rstFileReader.read();
}