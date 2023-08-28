#include "parameterFileReader.hpp"

#include "angleSection.hpp"              // for AngleSection
#include "bondSection.hpp"               // for BondSection
#include "dihedralSection.hpp"           // for DihedralSection
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for InputFileException, ParameterFileExce...
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "nonCoulombicsSection.hpp"      // for NonCoulombicsSection
#include "settings.hpp"                  // for Settings
#include "stringUtilities.hpp"           // for removeComments, splitString, toLowerCopy
#include "typesSection.hpp"              // for TypesSection

#include <filesystem>   // for exists
#include <functional>   // for identity
#include <ranges>       // for __find_if_fn, find_if

using namespace readInput::parameterFile;

/**
 * @brief constructor
 *
 * @details initializes file pointer _fp with filename and adds all parameter file sections to _parameterFileSections
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(const std::string &filename, engine::Engine &engine)
    : _fileName(filename), _fp(filename), _engine(engine)
{
    _parameterFileSections.push_back(std::make_unique<TypesSection>());
    _parameterFileSections.push_back(std::make_unique<BondSection>());
    _parameterFileSections.push_back(std::make_unique<AngleSection>());
    _parameterFileSections.push_back(std::make_unique<DihedralSection>());
    _parameterFileSections.push_back(std::make_unique<ImproperDihedralSection>());
    _parameterFileSections.push_back(std::make_unique<NonCoulombicsSection>());
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if force field is activated
 * @return false
 */
bool ParameterFileReader::isNeeded() const
{
    if (_engine.isForceFieldActivated())
        return true;

    return false;
}

/**
 * @brief determines which section of the parameter file the header line belongs to
 *
 * @param lineElements
 * @return ParameterFileSection*
 *
 * @throws customException::ParameterFileException if unknown or already parsed keyword
 */
ParameterFileSection *ParameterFileReader::determineSection(const std::vector<std::string> &lineElements)
{
    const auto iterEnd = _parameterFileSections.end();

    for (auto section = _parameterFileSections.begin(); section != iterEnd; ++section)
        if ((*section)->keyword() == utilities::toLowerCopy(lineElements[0]))
            return (*section).get();

    throw customException::ParameterFileException("Unknown or already parsed keyword \"" + lineElements[0] +
                                                  "\" in parameter file");
}

/**
 * @brief deletes section from _parameterFileSections
 *
 * @param section
 */
void ParameterFileReader::deleteSection(const ParameterFileSection *section)
{
    auto       sectionIsEqual = [section](auto &sectionUniquePtr) { return sectionUniquePtr.get() == section; };
    const auto result         = std::ranges::find_if(_parameterFileSections, sectionIsEqual);
    _parameterFileSections.erase(result);
}

/**
 * @brief reads parameter file
 *
 * @details Reads parameter file and according to the first word of the line, the corresponding section is called and then the
 * line is processed. After processing the line, the section is deleted from _parameterFileSections.
 *
 * @throws customException::InputFileException if file was not provided
 * @throws customException::InputFileException if file does not exist
 */
void ParameterFileReader::read()
{
    if (!isNeeded())
        return;

    if (_fileName.empty())
        throw customException::InputFileException("Parameter file needed for requested simulation setup");

    if (!std::filesystem::exists(_fileName))
        throw customException::InputFileException("Parameter file \"" + _fileName + "\"" + " File not found");

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

        auto *section = determineSection(lineElements);
        ++lineNumber;
        section->setLineNumber(lineNumber);
        section->setFp(&_fp);
        section->process(lineElements, _engine);
        lineNumber = section->getLineNumber();

        deleteSection(section);
    }
}

/**
 * @brief constructs a ParameterFileReader and reads parameter file
 *
 * @param filename
 * @param engine
 */
void readInput::parameterFile::readParameterFile(engine::Engine &engine)
{
    ParameterFileReader parameterFileReader(engine.getSettings().getParameterFilename(), engine);
    parameterFileReader.read();
}