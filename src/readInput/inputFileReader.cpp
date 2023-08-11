#include "inputFileReader.hpp"

#include "constants.hpp"
#include "inputFileParserCellList.hpp"
#include "inputFileParserConstraints.hpp"
#include "inputFileParserCoulombLongRange.hpp"
#include "inputFileParserForceField.hpp"
#include "inputFileParserGeneral.hpp"
#include "inputFileParserIntegrator.hpp"
#include "inputFileParserManostat.hpp"
#include "inputFileParserNonCoulomb.hpp"
#include "inputFileParserOutput.hpp"
#include "inputFileParserParameterFile.hpp"
#include "inputFileParserResetKinetics.hpp"
#include "inputFileParserSimulationBox.hpp"
#include "inputFileParserThermostat.hpp"
#include "inputFileParserTimings.hpp"
#include "inputFileParserTopology.hpp"
#include "inputFileParserVirial.hpp"
#include "stringUtilities.hpp"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace utilities;
using namespace readInput;
using namespace thermostat;
using namespace manostat;
using namespace engine;
using namespace customException;
using namespace resetKinetics;

/**
 * @brief Construct a new Input File Reader:: Input File Reader object
 *
 * @param filename
 * @param engine
 */
InputFileReader::InputFileReader(const std::string &filename, engine::Engine &engine) : _filename(filename), _engine(engine)
{
    _parsers.push_back(make_unique<InputFileParserCellList>(_engine));
    _parsers.push_back(make_unique<InputFileParserConstraints>(_engine));
    _parsers.push_back(make_unique<InputFileParserCoulombLongRange>(_engine));
    _parsers.push_back(make_unique<InputFileParserForceField>(_engine));
    _parsers.push_back(make_unique<InputFileParserGeneral>(_engine));
    _parsers.push_back(make_unique<InputFileParserIntegrator>(_engine));
    _parsers.push_back(make_unique<InputFileParserManostat>(_engine));
    _parsers.push_back(make_unique<InputFileParserNonCoulomb>(_engine));
    _parsers.push_back(make_unique<InputFileParserOutput>(_engine));
    _parsers.push_back(make_unique<InputFileParserParameterFile>(_engine));
    _parsers.push_back(make_unique<InputFileParserResetKinetics>(_engine));
    _parsers.push_back(make_unique<InputFileParserSimulationBox>(_engine));
    _parsers.push_back(make_unique<InputFileParserThermostat>(_engine));
    _parsers.push_back(make_unique<InputFileParserTimings>(_engine));
    _parsers.push_back(make_unique<InputFileParserTopology>(_engine));
    _parsers.push_back(make_unique<InputFileParserVirial>(_engine));

    addKeywords();
}

/**
 * @brief collects all the keywords from all the parsers
 *
 * @details inserts all keywords-std::function maps from all parsers into a single map
 * inserts all keywords-required maps from all parsers into a single map
 * inserts all keywords-count maps from all parsers into a single map
 *
 */
void InputFileReader::addKeywords()
{
    auto addKeyword = [&](const auto &parser)
    {
        const auto keywordRequiredMap = parser->getKeywordRequiredMap();
        const auto keywordFuncMap     = parser->getKeywordFuncMap();
        const auto keywordCountMap    = parser->getKeywordCountMap();

        _keywordRequiredMap.insert(keywordRequiredMap.begin(), keywordRequiredMap.end());
        _keywordFuncMap.insert(keywordFuncMap.begin(), keywordFuncMap.end());
        _keywordCountMap.insert(keywordCountMap.begin(), keywordCountMap.end());
    };

    ranges::for_each(_parsers, addKeyword);
}

/**
 * @brief process command
 *
 * @param lineElements
 *
 * @throw InputFileException if keyword is not recognised
 */
void InputFileReader::process(const vector<string> &lineElements)
{
    const auto keyword = toLowerCopy(lineElements[0]);

    if (!_keywordFuncMap.contains(keyword))
        throw InputFileException(format("Invalid keyword \"{}\" at line {}", keyword, _lineNumber));

    ParseFunc parserFunc = _keywordFuncMap[keyword];
    parserFunc(lineElements, _lineNumber);

    ++_keywordCountMap[keyword];
}

/**
 * @brief read input file
 *
 * @throw InputFileException if file not found
 */
void InputFileReader::read()
{
    ifstream inputFile(_filename);
    string   line;

    if (inputFile.fail())
        throw InputFileException("\"" + _filename + "\"" + " File not found");

    while (getline(inputFile, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        for (const auto lineCommands = getLineCommands(line, _lineNumber); const string &command : lineCommands)
        {
            const auto lineElements = splitString(command);
            if (lineElements.empty())
                continue;

            process(lineElements);
        }

        ++_lineNumber;
    }
}

/**
 * @brief reads input file and sets settings
 *
 * @param filename
 * @param engine
 *
 */
void readInput::readInputFile(const string &filename, Engine &engine)
{
    InputFileReader inputFileReader(filename, engine);
    inputFileReader.read();
    inputFileReader.postProcess();
}

/**
 * @brief checking keywords set in input file
 *
 * @throw InputFileException if keyword is required but not found
 */
void InputFileReader::postProcess()
{
    for (auto const &[keyword, count] : _keywordCountMap)
    {
        if (_keywordRequiredMap[keyword] && (0 == count))
            throw InputFileException("Missing keyword \"" + keyword + "\" in input file");

        if (count > 1)
            throw InputFileException("Multiple keywords \"" + keyword + "\" in input file");
    }

    _engine.getSettings().setMoldescriptorFilename(_engine.getSettings().getGuffPath() + "/" +
                                                   _engine.getSettings().getMoldescriptorFilename());
}