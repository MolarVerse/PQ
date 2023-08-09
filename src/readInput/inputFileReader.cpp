#include "inputFileReader.hpp"

#include "constants.hpp"
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
 * @brief check if second argument is "="
 *
 * @param lineElement
 * @param _lineNumber
 *
 * @throw InputFileException if second argument is not "="
 */
void readInput::checkEqualSign(const string_view &lineElement, const size_t lineNumber)
{
    if (lineElement != "=") throw InputFileException("Invalid command at line " + to_string(lineNumber) + "in input file");
}

/**
 * @brief check if command array has at least 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less than 3 elements
 *
 * @note this function is used for commands that have an array as their third argument
 */
void readInput::checkCommandArray(const vector<string> &lineElements, const size_t lineNumber)
{
    if (lineElements.size() < 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(lineNumber) + "in input file");

    checkEqualSign(lineElements[1], lineNumber);
}

/**
 * @brief check if command array has exactly 3 elements
 *
 * @param lineElements
 * @param _lineNumber
 *
 * @throw InputFileException if command array has less or more than 3 elements
 */
void readInput::checkCommand(const vector<string> &lineElements, const size_t lineNumber)
{
    if (lineElements.size() != 3)
        throw InputFileException("Invalid number of arguments at line " + to_string(lineNumber) + "in input file");

    checkEqualSign(lineElements[1], lineNumber);
}

/**
 * @brief Construct a new Input File Reader:: Input File Reader object
 *
 * @param filename
 * @param settings
 *
 * @details parsing functions stored in a keyword map as function pointers
 */
InputFileReader::InputFileReader(const string &filename, Engine &engine) : _filename(filename), _engine(engine)
{
    addKeyword(string("jobtype"), bind_front(&InputFileReader::parseJobType, this), true);

    addKeyword(string("timestep"), bind_front(&InputFileReader::parseTimestep, this), true);
    addKeyword(string("nstep"), bind_front(&InputFileReader::parseNumberOfSteps, this), true);

    addKeyword(string("start_file"), bind_front(&InputFileReader::parseStartFilename, this), true);
    addKeyword(string("moldescriptor_file"), bind_front(&InputFileReader::parseMoldescriptorFilename, this), false);
    addKeyword(string("guff_path"), bind_front(&InputFileReader::parseGuffPath, this), false);
    addKeyword(string("guff_file"), bind_front(&InputFileReader::parseGuffDatFilename, this), false);

    addKeyword(string("output_freq"), bind_front(&InputFileReader::parseOutputFreq, this), false);
    addKeyword(string("output_file"), bind_front(&InputFileReader::parseLogFilename, this), false);
    addKeyword(string("info_file"), bind_front(&InputFileReader::parseInfoFilename, this), false);
    addKeyword(string("energy_file"), bind_front(&InputFileReader::parseEnergyFilename, this), false);
    addKeyword(string("traj_file"), bind_front(&InputFileReader::parseTrajectoryFilename, this), false);
    addKeyword(string("vel_file"), bind_front(&InputFileReader::parseVelocityFilename, this), false);
    addKeyword(string("force_file"), bind_front(&InputFileReader::parseForceFilename, this), false);
    addKeyword(string("restart_file"), bind_front(&InputFileReader::parseRestartFilename, this), false);
    addKeyword(string("charge_file"), bind_front(&InputFileReader::parseChargeFilename, this), false);

    addKeyword(string("integrator"), bind_front(&InputFileReader::parseIntegrator, this), true);

    addKeyword(string("density"), bind_front(&InputFileReader::parseDensity, this), false);

    addKeyword(string("virial"), bind_front(&InputFileReader::parseVirial, this), false);

    addKeyword(string("rcoulomb"), bind_front(&InputFileReader::parseCoulombRadius, this), false);
    addKeyword(string("long_range"), bind_front(&InputFileReader::parseCoulombLongRange, this), false);
    addKeyword(string("wolf_param"), bind_front(&InputFileReader::parseWolfParameter, this), false);

    addKeyword(string("noncoulomb"), bind_front(&InputFileReader::parseNonCoulombType, this), false);

    addKeyword(string("cell-list"), bind_front(&InputFileReader::parseCellListActivated, this), false);
    addKeyword(string("cell-number"), bind_front(&InputFileReader::parseNumberOfCells, this), false);

    addKeyword(string("thermostat"), bind_front(&InputFileReader::parseThermostat, this), false);
    addKeyword(string("temp"), bind_front(&InputFileReader::parseTemperature, this), false);
    addKeyword(string("t_relaxation"), bind_front(&InputFileReader::parseThermostatRelaxationTime, this), false);

    addKeyword(string("manostat"), bind_front(&InputFileReader::parseManostat, this), false);
    addKeyword(string("pressure"), bind_front(&InputFileReader::parsePressure, this), false);
    addKeyword(string("p_relaxation"), bind_front(&InputFileReader::parseManostatRelaxationTime, this), false);
    addKeyword(string("compressibility"), bind_front(&InputFileReader::parseCompressibility, this), false);

    addKeyword(string("nscale"), bind_front(&InputFileReader::parseNScale, this), false);
    addKeyword(string("fscale"), bind_front(&InputFileReader::parseFScale, this), false);
    addKeyword(string("nreset"), bind_front(&InputFileReader::parseNReset, this), false);
    addKeyword(string("freset"), bind_front(&InputFileReader::parseFReset, this), false);

    addKeyword(string("shake"), bind_front(&InputFileReader::parseShakeActivated, this), false);
    addKeyword(string("shake-tolerance"), bind_front(&InputFileReader::parseShakeTolerance, this), false);
    addKeyword(string("shake-iter"), bind_front(&InputFileReader::parseShakeIteration, this), false);
    addKeyword(string("rattle-iter"), bind_front(&InputFileReader::parseRattleIteration, this), false);
    addKeyword(string("rattle-tolerance"), bind_front(&InputFileReader::parseRattleTolerance, this), false);

    addKeyword(string("topology_file"), bind_front(&InputFileReader::parseTopologyFilename, this), false);
    addKeyword(string("parameter_file"), bind_front(&InputFileReader::parseParameterFilename, this), false);

    addKeyword(string("force-field"), bind_front(&InputFileReader::parseForceFieldType, this), false);
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
void InputFileReader::addKeyword(const string &keyword, ParseFunc parserFunc, bool required)
{
    _keywordFuncMap.try_emplace(keyword, parserFunc);
    _keywordCountMap.try_emplace(keyword, 0);
    _keywordRequiredMap.try_emplace(keyword, required);
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
    const auto keyword = boost::algorithm::to_lower_copy(lineElements[0]);

    if (!_keywordFuncMap.contains(keyword))
        throw InputFileException("Invalid keyword \"" + keyword + "\" at line " + to_string(_lineNumber));

    ParseFunc parserFunc = _keywordFuncMap[keyword];
    parserFunc(lineElements);

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

    if (inputFile.fail()) throw InputFileException("\"" + _filename + "\"" + " File not found");

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
            if (lineElements.empty()) continue;

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

        if (count > 1) throw InputFileException("Multiple keywords \"" + keyword + "\" in input file");
    }

    _engine.getSettings().setMoldescriptorFilename(_engine.getSettings().getGuffPath() + "/" +
                                                   _engine.getSettings().getMoldescriptorFilename());
}