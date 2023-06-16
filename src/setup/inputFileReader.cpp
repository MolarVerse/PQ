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
using namespace StringUtilities;
using namespace setup;
using namespace thermostat;

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
    addKeyword(string("jobtype"), &InputFileReader::parseJobType, true);

    addKeyword(string("timestep"), &InputFileReader::parseTimestep, true);
    addKeyword(string("nstep"), &InputFileReader::parseNumberOfSteps, true);

    addKeyword(string("start_file"), &InputFileReader::parseStartFilename, true);
    addKeyword(string("moldescriptor_file"), &InputFileReader::parseMoldescriptorFilename, false);
    addKeyword(
        string("guff_path"), &InputFileReader::parseGuffPath, false);   // default is current directory (not backword compatible)

    addKeyword(string("output_freq"), &InputFileReader::parseOutputFreq, false);
    addKeyword(string("output_file"), &InputFileReader::parseLogFilename, false);
    addKeyword(string("info_file"), &InputFileReader::parseInfoFilename, false);
    addKeyword(string("energy_file"), &InputFileReader::parseEnergyFilename, false);
    addKeyword(string("traj_file"), &InputFileReader::parseTrajectoryFilename, false);
    addKeyword(string("vel_file"), &InputFileReader::parseVelocityFilename, false);
    addKeyword(string("force_file"), &InputFileReader::parseForceFilename, false);
    addKeyword(string("restart_file"), &InputFileReader::parseRestartFilename, false);
    addKeyword(string("charge_file"), &InputFileReader::parseChargeFilename, false);

    addKeyword(string("integrator"), &InputFileReader::parseIntegrator, true);

    addKeyword(string("density"), &InputFileReader::parseDensity, false);

    addKeyword(string("virial"), &InputFileReader::parseVirial, false);

    addKeyword(string("rcoulomb"), &InputFileReader::parseRcoulomb, false);

    addKeyword(string("cell-list"), &InputFileReader::parseCellListActivated, false);
    addKeyword(string("cell-number"), &InputFileReader::parseNumberOfCells, false);

    addKeyword(string("thermostat"), &InputFileReader::parseThermostat, false);
    addKeyword(string("temp"), &InputFileReader::parseTemperature, false);
    addKeyword(string("t_relaxation"), &InputFileReader::parseThermostatRelaxationTime, false);

    addKeyword(string("manostat"), &InputFileReader::parseManostat, false);
    addKeyword(string("pressure"), &InputFileReader::parsePressure, false);
    addKeyword(string("p_relaxation"), &InputFileReader::parseManostatRelaxationTime, false);
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
void InputFileReader::addKeyword(const string &keyword,
                                 void (InputFileReader::*parserFunc)(const vector<string> &),
                                 bool required)
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

    if (_keywordFuncMap.find(keyword) == _keywordFuncMap.end())
        throw InputFileException("Invalid keyword \"" + keyword + "\" at line " + to_string(_lineNumber));

    void (InputFileReader::*parserFunc)(const vector<string> &) = _keywordFuncMap[keyword];
    (this->*parserFunc)(lineElements);

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

        const auto lineCommands = getLineCommands(line, _lineNumber);

        for (const string &command : lineCommands)
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
void setup::readInputFile(const string &filename, Engine &engine)
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
        if (_keywordRequiredMap[keyword] && (count == 0))
            throw InputFileException("Missing keyword \"" + keyword + "\" in input file");

        if (count > 1) throw InputFileException("Multiple keywords \"" + keyword + "\" in input file");
    }

    _engine.getSettings().setMoldescriptorFilename(_engine.getSettings().getGuffPath() + "/" +
                                                   _engine.getSettings().getMoldescriptorFilename());

    setupThermostat();
    setupManostat();
}

void InputFileReader::setupThermostat()   // TODO: include warnings if value set but not used
{
    if (_engine.getSettings().getThermostat() == "berendsen")
    {
        if (!_engine.getSettings().getTemperatureSet()) throw InputFileException("Temperature not set for Berendsen thermostat");

        if (!_engine.getSettings().getRelaxationTimeSet())
        {
            _engine._stdoutOutput->writeRelaxationTimeThermostatWarning();
            _engine._logOutput->writeRelaxationTimeThermostatWarning();
        }

        _engine._thermostat = make_unique<BerendsenThermostat>(_engine.getSettings().getTemperature(),
                                                               _engine.getSettings().getRelaxationTime() * _PS_TO_FS_);
    }
    else
    {
        // warnings if values set but not used
    }

    _engine._thermostat->setTimestep(_engine.getTimings().getTimestep());
}

void InputFileReader::setupManostat()   // TODO: include warnings if value set but not used
{
    if (_engine.getSettings().getManostat() == "berendsen")
    {
        if (!_engine.getSettings().getManostatSet()) throw InputFileException("Pressure not set for Berendsen manostat");

        if (!_engine.getSettings().getTauManostatSet())
        {
            _engine._stdoutOutput->writeRelaxationTimeManostatWarning();
            _engine._logOutput->writeRelaxationTimeManostatWarning();
        }

        _engine._manostat = make_unique<BerendsenManostat>(_engine.getSettings().getPressure(),
                                                           _engine.getSettings().getTauManostat() * _PS_TO_FS_);
    }
    else
    {
        // warnings if values set but not used
    }

    _engine._manostat->setTimestep(_engine.getTimings().getTimestep());
}