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
    _keywordMap.try_emplace(string("jobtype"), &InputFileReader::parseJobType);

    _keywordMap.try_emplace(string("timestep"), &InputFileReader::parseTimestep);
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

    if (_keywordMap.find(keyword) == _keywordMap.end())
        throw runtime_error("Invalid keyword \"" + keyword + "\" at line " + to_string(_lineNumber));

    void (InputFileReader::*parserFunc)(const vector<string> &) = _keywordMap[keyword];
    (this->*parserFunc)(lineElements);
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
}