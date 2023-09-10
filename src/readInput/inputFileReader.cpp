#include "inputFileReader.hpp"

#include "engine.hpp"                            // for Engine
#include "exceptions.hpp"                        // for InputFileException
#include "inputFileParserCellList.hpp"           // for InputFileParserCellList
#include "inputFileParserConstraints.hpp"        // for InputFileParserConstraints
#include "inputFileParserCoulombLongRange.hpp"   // for InputFileParserCoulombLongRange
#include "inputFileParserFiles.hpp"              // for InputFileParserFiles
#include "inputFileParserForceField.hpp"         // for InputFileParserForceField
#include "inputFileParserGeneral.hpp"            // for InputFileParserGeneral
#include "inputFileParserIntegrator.hpp"         // for InputFileParserIntegrator
#include "inputFileParserManostat.hpp"           // for InputFileParserManostat
#include "inputFileParserNonCoulomb.hpp"         // for InputFileParserNonCoulomb
#include "inputFileParserOutput.hpp"             // for InputFileParserOutput
#include "inputFileParserResetKinetics.hpp"      // for InputFileParserResetKinetics
#include "inputFileParserSimulationBox.hpp"      // for InputFileParserSimulationBox
#include "inputFileParserThermostat.hpp"         // for InputFileParserThermostat
#include "inputFileParserTimings.hpp"            // for InputFileParserTimings
#include "inputFileParserVirial.hpp"             // for InputFileParserVirial
#include "stringUtilities.hpp"                   // for getLineCommands, removeComments, splitString, toLowerCopy

#include <algorithm>   // for __for_each_fn, for_each
#include <format>      // for format
#include <fstream>     // for ifstream, basic_istream
#include <map>         // for map, operator==
#include <string>      // for char_traits, string
#include <vector>      // for vector

using namespace readInput;

/**
 * @brief Construct a new Input File Reader:: Input File Reader object
 *
 * @details adds all parsers to the _parsers vector and calls addKeywords() to add all keywords to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap
 *
 * @param fileName
 * @param engine
 */
InputFileReader::InputFileReader(const std::string_view &fileName, engine::Engine &engine) : _fileName(fileName), _engine(engine)
{
    _parsers.push_back(std::make_unique<InputFileParserCellList>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserConstraints>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserCoulombLongRange>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserFiles>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserForceField>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserGeneral>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserIntegrator>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserManostat>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserNonCoulomb>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserOutput>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserResetKinetics>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserSimulationBox>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserThermostat>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserTimings>(_engine));
    _parsers.push_back(std::make_unique<InputFileParserVirial>(_engine));

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

    std::ranges::for_each(_parsers, addKeyword);
}

/**
 * @brief process command
 *
 * @details Checks if keyword is in _keywordFuncMap, calls the corresponding function and increments the keyword count.
 *
 * @param lineElements
 *
 * @throw InputFileException if keyword is not recognised
 */
void InputFileReader::process(const std::vector<std::string> &lineElements)
{
    const auto keyword = utilities::toLowerCopy(lineElements[0]);

    if (!_keywordFuncMap.contains(keyword))
        throw customException::InputFileException(std::format("Invalid keyword \"{}\" at line {}", keyword, _lineNumber));

    ParseFunc parserFunc = _keywordFuncMap[keyword];
    parserFunc(lineElements, _lineNumber);

    ++_keywordCountMap[keyword];
}

/**
 * @brief read input file
 *
 * @details Reads input file line by line. One line can consist of multiple commands separated by semicolons. For each command the
 * process() function is called.
 *
 * @note Also single command lines have to be terminated with a semicolon. '#' is used for comments as in all other file formats.
 *
 * @throw InputFileException if file not found
 */
void InputFileReader::read()
{
    std::ifstream inputFile(_fileName);

    if (inputFile.fail())
        throw customException::InputFileException("\"" + _fileName + "\"" + " File not found");

    std::string line;

    while (getline(inputFile, line))
    {
        line = utilities::removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        auto processInputCommand = [this](const auto &command)
        {
            const auto lineElements = utilities::splitString(command);
            if (!lineElements.empty())
                process(lineElements);
        };

        std::ranges::for_each(utilities::getLineCommands(line, _lineNumber), processInputCommand);

        ++_lineNumber;
    }
}

void readInput::readJobType(const std::string &fileName, std::unique_ptr<engine::Engine> &engine)
{
    std::ifstream inputFile(fileName);

    if (inputFile.fail())
        throw customException::InputFileException("\"" + fileName + "\"" + " File not found");

    std::string line;
    size_t      lineNumber(1);
    bool        jobtypeFound{false};

    while (getline(inputFile, line))
    {
        line = utilities::removeComments(line, "#");

        if (line.empty())
        {
            ++lineNumber;
            continue;
        }

        auto processInputCommand = [lineNumber, &jobtypeFound, &engine](const auto &command)
        {
            const auto lineElements = utilities::splitString(command);
            if (!lineElements.empty() && "jobtype" == lineElements[0])
            {
                auto parser = InputFileParserGeneral(*engine);
                parser.parseJobTypeForEngine(lineElements, lineNumber, engine);
                jobtypeFound = true;
            }
        };

        std::ranges::for_each(utilities::getLineCommands(line, lineNumber), processInputCommand);

        ++lineNumber;
    }

    if (!jobtypeFound)
        throw customException::InputFileException("Missing keyword \"jobtype\" in input file");
}

/**
 * @brief wrapper function to construct InputFileReader and call read() and postProcess()
 *
 * @param fileName
 * @param engine
 *
 */
void readInput::readInputFile(const std::string_view &fileName, engine::Engine &engine)
{
    InputFileReader inputFileReader(fileName, engine);
    inputFileReader.read();
    inputFileReader.postProcess();
}

/**
 * @brief checking keywords set in input file and collects
 *
 * @throw InputFileException if keyword is required but not found
 * @throw InputFileException if keyword is found multiple times
 */
void InputFileReader::postProcess()
{
    auto checkKeyWordCount = [this](const auto &keyWordCountElement)
    {
        const auto &[keyword, count] = keyWordCountElement;

        if (_keywordRequiredMap[keyword] && (0 == count))
            throw customException::InputFileException("Missing keyword \"" + keyword + "\" in input file");

        if (count > 1)
            throw customException::InputFileException("Multiple keywords \"" + keyword + "\" in input file");
    };

    std::ranges::for_each(_keywordCountMap, checkKeyWordCount);
}