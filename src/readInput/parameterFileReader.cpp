#include "parameterFileReader.hpp"

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, ParameterFileExce...
#include "settings.hpp"     // for Settings
#include "stringUtilities.hpp"

#include <algorithm>    // for ranges::find_if
#include <filesystem>   // for exists
#include <functional>   // for identity

using namespace std;
using namespace readInput::parameterFile;
using namespace utilities;

/**
 * @brief constructor
 *
 * @details initializes file pointer _fp with filename and adds all parameter file sections to _parameterFileSections
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(const string &filename, engine::Engine &engine)
    : _filename(filename), _fp(filename), _engine(engine)
{
    _parameterFileSections.push_back(make_unique<TypesSection>());
    _parameterFileSections.push_back(make_unique<BondSection>());
    _parameterFileSections.push_back(make_unique<AngleSection>());
    _parameterFileSections.push_back(make_unique<DihedralSection>());
    _parameterFileSections.push_back(make_unique<ImproperDihedralSection>());
    _parameterFileSections.push_back(make_unique<NonCoulombicsSection>());
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if shake is activated
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
 */
ParameterFileSection *ParameterFileReader::determineSection(const std::vector<std::string> &lineElements)
{
    const auto iterEnd = _parameterFileSections.end();

    for (auto section = _parameterFileSections.begin(); section != iterEnd; ++section)
        if ((*section)->keyword() == toLowerCopy(lineElements[0]))
            return (*section).get();

    throw customException::ParameterFileException("Unknown or already passed keyword \"" + lineElements[0] +
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
    const auto result         = ranges::find_if(_parameterFileSections, sectionIsEqual);
    _parameterFileSections.erase(result);
}

/**
 * @brief reads parameter file
 */
void ParameterFileReader::read()
{
    string         line;
    vector<string> lineElements;
    int            lineNumber = 1;

    if (!isNeeded())
        return;

    if (_filename.empty())
        throw customException::InputFileException("Parameter file needed for requested simulation setup");

    if (!filesystem::exists(_filename))
        throw customException::InputFileException("Parameter file \"" + _filename + "\"" + " File not found");

    while (getline(_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

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