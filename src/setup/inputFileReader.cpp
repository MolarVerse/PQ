#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <iostream>

#include "inputFileReader.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace StringUtilities;
using namespace Setup::InputFileReader;

/**
 * @brief Construct a new Input File Reader:: Input File Reader object
 *
 * @param filename
 * @param settings
 *
 * @details parsing functions stored in a keyword map as function pointers
 */
InputFileReader::InputFileReader(const string &filename, Settings &settings) : _filename(filename), _settings(settings)
{
    addKeyword(string("jobtype"), &InputFileReader::parseJobType, true);

    addKeyword(string("timestep"), &InputFileReader::parseTimestep, true);
    addKeyword(string("nstep"), &InputFileReader::parseNumberOfSteps, true);

    addKeyword(string("start_file"), &InputFileReader::parseStartFilename, true);

    addKeyword(string("output_freq"), &InputFileReader::parseOutputFreq, false);
    addKeyword(string("output_file"), &InputFileReader::parseLogFilename, false);
}

/**
 * @brief add keyword to different keyword maps
 *
 * @param keyword
 * @param parserFunc
 * @param count
 * @param required
 *
 * @details
 *
 *  parserFunc is a function pointer to a parsing function
 *  count is the number of keywords found in the inputfile
 *  required is a boolean that indicates if the keyword is required
 *
 */
void InputFileReader::addKeyword(const string &keyword, void (InputFileReader::*parserFunc)(const vector<string> &), bool required)
{
    _keywordFuncMap.try_emplace(keyword, parserFunc);
    _keywordCountMap.try_emplace(keyword, 0);
    _keywordRequiredMap.try_emplace(keyword, required);
}

/**
 * @brief get commands from a line
 *
 * @param line
 * @return vector<string>
 *
 * @throw invalid_argument if line does not end with a semicolon
 */
vector<string> getLineCommands(const string &line, int _lineNumber)
{

    for (int i = int(line.size()) - 1; i >= 0; i--)
    {
        if (line[i] == ';')
            break;
        else if (!isspace(line[i]))
            throw invalid_argument("Missing semicolon in input file at line " + to_string(_lineNumber));
    }

    vector<string> lineCommands;
    boost::split(lineCommands, line, boost::is_any_of(";"));

    return lineCommands;
}

/**
 * @brief process command
 *
 * @param lineElements
 *
 * @throw runtime_error if keyword is not recognised
 */
void InputFileReader::process(const vector<string> &lineElements)
{
    auto keyword = boost::algorithm::to_lower_copy(lineElements[0]);

    if (_keywordFuncMap.find(keyword) == _keywordFuncMap.end())
        throw runtime_error("Invalid keyword \"" + keyword + "\" at line " + to_string(_lineNumber));

    void (InputFileReader::*parserFunc)(const vector<string> &) = _keywordFuncMap[keyword];
    (this->*parserFunc)(lineElements);

    _keywordCountMap[keyword]++;
}

/**
 * @brief read input file
 *
 * @throw runtime_error if file not found
 */
void InputFileReader::read()
{
    ifstream inputFile(_filename);
    string line;

    if (inputFile.fail())
        throw runtime_error("\"" + _filename + "\"" + " File not found");

    while (getline(inputFile, line))
    {
        line = removeComments(line, "#");
        auto lineCommands = getLineCommands(line, _lineNumber);

        for (const string &command : lineCommands)
        {
            auto lineElements = splitString(command);
            if (lineElements.empty())
                continue;

            process(lineElements);
        }

        _lineNumber++;
    }
}

/**
 * @brief read input file and instantiate InputFileReader
 *
 * @param filename
 * @param settings
 */
void readInputFile(const string &filename, Settings &settings)
{
    InputFileReader inputFileReader(filename, settings);
    inputFileReader.read();
    inputFileReader.postProcess();
}

/**
 * @brief checking keywords set in input file
 *
 */
void InputFileReader::postProcess()
{
    for (auto const &[keyword, count] : _keywordCountMap)
    {
        if (_keywordRequiredMap[keyword] && count == 0)
            throw runtime_error("Missing keyword \"" + keyword + "\" in input file");

        if (count > 1)
            throw runtime_error("Multiple keywords \"" + keyword + "\" in input file");
    }
}