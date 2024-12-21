/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "inputFileReader.hpp"

#include <algorithm>   // for __for_each_fn, for_each
#include <format>      // for format
#include <fstream>     // for ifstream, basic_istream
#include <map>         // for map, operator==
#include <string>      // for char_traits, string
#include <vector>      // for vector

#include "QMInputParser.hpp"                 // for InputFileParserQM
#include "cellListInputParser.hpp"           // for CellListInputParser
#include "constraintsInputParser.hpp"        // for InputFileParserConstraints
#include "convergenceInputParser.hpp"        // for ConvergenceInputParser
#include "coulombLongRangeInputParser.hpp"   // for InputFileParserCoulombLongRange
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for InputFileException
#include "filesInputParser.hpp"              // for InputFileParserFiles
#include "forceFieldInputParser.hpp"         // for InputFileParserForceField
#include "generalInputParser.hpp"            // for InputFileParserGeneral
#include "hybridInputParser.hpp"             // for InputFileParserQMMM
#include "integratorInputParser.hpp"         // for InputFileParserIntegrator
#include "manostatInputParser.hpp"           // for InputFileParserManostat
#include "nonCoulombInputParser.hpp"         // for InputFileParserNonCoulomb
#include "optInputParser.hpp"                // for OptInputParser
#include "outputInputParser.hpp"             // for InputFileParserOutput
#include "resetKineticsInputParser.hpp"      // for InputFileParserResetKinetics
#include "ringPolymerInputParser.hpp"        // for InputFileParserRingPolymer
#include "simulationBoxInputParser.hpp"      // for InputFileParserSimulationBox
#include "stringUtilities.hpp"         // for getLineCommands, removeComments
#include "thermostatInputParser.hpp"   // for InputFileParserThermostat
#include "timingsInputParser.hpp"      // for InputFileParserTimings
#include "virialInputParser.hpp"       // for InputFileParserVirial

using namespace input;
using namespace utilities;
using namespace customException;
using std::make_unique;

/**
 * @brief Construct a new Input File Reader:: Input File Reader object
 *
 * @details adds all parsers to the _parsers vector and calls addKeywords() to
 * add all keywords to the _keywordFuncMap, _keywordRequiredMap and
 * _keywordCountMap
 *
 * @param fileName
 * @param engine
 */
InputFileReader::InputFileReader(
    const std::string_view &fileName,
    engine::Engine         &engine
)
    : _fileName(fileName), _engine(engine)
{
    _parsers.push_back(make_unique<CellListInputParser>(_engine));
    _parsers.push_back(make_unique<ConstraintsInputParser>(_engine));
    _parsers.push_back(make_unique<CoulombLongRangeInputParser>(_engine));
    _parsers.push_back(make_unique<FilesInputParser>(_engine));
    _parsers.push_back(make_unique<ForceFieldInputParser>(_engine));
    _parsers.push_back(make_unique<GeneralInputParser>(_engine));
    _parsers.push_back(make_unique<IntegratorInputParser>(_engine));
    _parsers.push_back(make_unique<ManostatInputParser>(_engine));
    _parsers.push_back(make_unique<NonCoulombInputParser>(_engine));
    _parsers.push_back(make_unique<OutputInputParser>(_engine));
    _parsers.push_back(make_unique<ResetKineticsInputParser>(_engine));
    _parsers.push_back(make_unique<SimulationBoxInputParser>(_engine));
    _parsers.push_back(make_unique<ThermostatInputParser>(_engine));
    _parsers.push_back(make_unique<TimingsInputParser>(_engine));
    _parsers.push_back(make_unique<VirialInputParser>(_engine));
    _parsers.push_back(make_unique<HybridInputParser>(_engine));
    _parsers.push_back(make_unique<RingPolymerInputParser>(_engine));

    _parsers.push_back(make_unique<ConvInputParser>(_engine));
    _parsers.push_back(make_unique<OptInputParser>(_engine));
    _parsers.push_back(make_unique<QMInputParser>(_engine));

    addKeywords();
}

/**
 * @brief collects all the keywords from all the parsers
 *
 * @details inserts all keywords-std::function maps from all parsers into a
 * single map inserts all keywords-required maps from all parsers into a single
 * map inserts all keywords-count maps from all parsers into a single map
 *
 */
void InputFileReader::addKeywords()
{
    auto addKeyword = [&](const auto &parser)
    {
        const auto keywordRequiredMap = parser->getKeywordRequiredMap();
        const auto keywordFuncMap     = parser->getKeywordFuncMap();
        const auto keywordCountMap    = parser->getKeywordCountMap();

        _keywordRequiredMap.insert(
            keywordRequiredMap.begin(),
            keywordRequiredMap.end()
        );

        _keywordFuncMap.insert(keywordFuncMap.begin(), keywordFuncMap.end());
        _keywordCountMap.insert(keywordCountMap.begin(), keywordCountMap.end());
    };

    std::ranges::for_each(_parsers, addKeyword);
}

/**
 * @brief process command
 *
 * @details Checks if keyword is in _keywordFuncMap, calls the corresponding
 * function and increments the keyword count.
 *
 * @param lineElements
 *
 * @throw InputFileException if keyword is not recognised
 */
void InputFileReader::process(const std::vector<std::string> &lineElements)
{
    const auto keyword = toLowerAndReplaceDashesCopy(lineElements[0]);

    if (!_keywordFuncMap.contains(keyword))
        throw InputFileException(std::format(
            "Invalid keyword \"{}\" at line {}",
            keyword,
            _lineNumber
        ));

    pq::ParseFunc parserFunc = _keywordFuncMap[keyword];
    parserFunc(lineElements, _lineNumber);

    ++_keywordCountMap[keyword];
}

/**
 * @brief read input file
 *
 * @details Reads input file line by line. One line can consist of multiple
 * commands separated by semicolons. For each command the process() function is
 * called.
 *
 * @note Also single command lines have to be terminated with a semicolon. '#'
 * is used for comments as in all other file formats.
 *
 * @throw InputFileException if file not found
 */
void InputFileReader::read()
{
    std::ifstream inputFile(_fileName);

    if (inputFile.fail())
        throw InputFileException("\"" + _fileName + "\"" + " File not found");

    std::string line;

    while (getline(inputFile, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        auto processInputCommand = [this](auto &command)
        {
            processEqualSign(command, _lineNumber);

            const auto lineElements = splitString(command);
            if (!lineElements.empty())
                process(lineElements);
        };

        std::ranges::for_each(
            getLineCommands(line, _lineNumber),
            processInputCommand
        );

        ++_lineNumber;
    }
}

/**
 * @brief checks if in the input file jobtype keyword is set and calls the
 * corresponding parser
 *
 * @details this is just the first parsing of the input file and includes only
 * the jobtype keyword
 *
 * @param fileName
 * @param engine
 */
void input::readJobType(
    const std::string               &fileName,
    std::unique_ptr<engine::Engine> &engine
)
{
    std::ifstream inputFile(fileName);

    if (inputFile.fail())
        throw InputFileException("\"" + fileName + "\"" + " File not found");

    std::string line;
    size_t      lineNumber(1);
    bool        jobtypeFound{false};

    while (getline(inputFile, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            ++lineNumber;
            continue;
        }

        auto processInputCommand =
            [lineNumber, &jobtypeFound, &engine](auto &command)
        {
            processEqualSign(command, lineNumber);

            const auto lineElements = splitString(command);
            if (!lineElements.empty() && "jobtype" == lineElements[0])
            {
                auto parser = GeneralInputParser(*engine);
                parser.parseJobTypeForEngine(lineElements, lineNumber, engine);
                jobtypeFound = true;
            }
        };

        std::ranges::for_each(
            getLineCommands(line, lineNumber),
            processInputCommand
        );

        ++lineNumber;
    }

    if (!jobtypeFound)
        throw InputFileException("Missing keyword \"jobtype\" in input file");
}

/**
 * @brief wrapper function to construct InputFileReader and call read() and
 * postProcess()
 *
 * @param fileName
 * @param engine
 *
 */
void input::readInputFile(
    const std::string_view &fileName,
    engine::Engine         &engine
)
{
    engine.getStdoutOutput().writeRead("Input File", std::string(fileName));

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
            throw InputFileException(
                "Missing keyword \"" + keyword + "\" in input file"
            );

        if (count > 1)
            throw InputFileException(
                "Multiple keywords \"" + keyword + "\" in input file"
            );
    };

    std::ranges::for_each(_keywordCountMap, checkKeyWordCount);
}

/**
 * @brief process equal sign
 *
 * @details replaces equal sign with " = " to make sure that the equal sign is
 * always surrounded by spaces
 *
 * @param command
 * @param lineNumber
 *
 * @throw InputFileException if equal sign is missing
 */
void input::processEqualSign(std::string &command, const size_t lineNumber)
{
    const auto equalSignPos = command.find('=');
    if (equalSignPos != std::string::npos)
        command.replace(equalSignPos, 1, " = ");

    else
        throw InputFileException(std::format(
            "Missing equal sign in command \"{}\" in line {}",
            command,
            lineNumber
        ));
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the input file name
 *
 * @param fileName
 */
void InputFileReader::setFilename(const std::string_view fileName)
{
    _fileName = fileName;
}

/**
 * @brief sets the keyword count
 *
 * @param keyword
 * @param count
 */
void InputFileReader::setKeywordCount(
    const std::string &keyword,
    const size_t       count
)
{
    _keywordCountMap[keyword] = count;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the keyword count
 *
 * @param keyword
 * @return size_t
 */
size_t InputFileReader::getKeywordCount(const std::string &keyword)
{
    return _keywordCountMap[keyword];
}

/**
 * @brief get the keyword required
 *
 * @param keyword
 * @return bool
 */
bool InputFileReader::getKeywordRequired(const std::string &keyword)
{
    return _keywordRequiredMap[keyword];
}

/**
 * @brief get the keyword count map
 *
 * @return std::map<std::string, size_t>
 */
std::map<std::string, size_t> InputFileReader::getKeywordCountMap() const
{
    return _keywordCountMap;
}

/**
 * @brief get the keyword required map
 *
 * @return std::map<std::string, bool>
 */
std::map<std::string, bool> InputFileReader::getKeywordRequiredMap() const
{
    return _keywordRequiredMap;
}

/**
 * @brief get the keyword function map
 *
 * @return std::map<std::string, pq::ParseFunc>
 */
std::map<std::string, pq::ParseFunc> InputFileReader::getKeywordFuncMap() const
{
    return _keywordFuncMap;
}