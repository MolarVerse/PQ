#ifndef _INPUT_FILE_READER_HPP_

#define _INPUT_FILE_READER_HPP_

#include "engine.hpp"
#include "exceptions.hpp"
#include "output.hpp"

#include <map>
#include <string>
#include <vector>

/**
 * @brief namespace for reading input files
 *
 */
namespace readInput
{
    class InputFileReader;
    void readInputFile(const std::string &, engine::Engine &);

    void checkEqualSign(const std::string_view &view, const size_t);
    void checkCommand(const std::vector<std::string> &, const size_t);
    void checkCommandArray(const std::vector<std::string> &, const size_t);

}   // namespace readInput

using ParseFunc = void (readInput::InputFileReader::*)(const std::vector<std::string> &);

/**
 * @class InputFileReader
 *
 * @brief reads input file and sets settings
 *
 */
class readInput::InputFileReader
{
  private:
    std::string     _filename;
    engine::Engine &_engine;

    std::map<std::string, ParseFunc> _keywordFuncMap;
    std::map<std::string, int>       _keywordCountMap;
    std::map<std::string, bool>      _keywordRequiredMap;

    size_t _lineNumber = 1;

  public:
    InputFileReader(const std::string &, engine::Engine &);

    void read();
    void addKeyword(const std::string &, ParseFunc, bool);
    void process(const std::vector<std::string> &);
    void postProcess();

    /*************************
     *                       *
     * all parsing functions *
     *                       *
     *************************/

    // parsing jobtype related keywords
    void parseJobType(const std::vector<std::string> &);

    // parsing timings related keywords
    void parseTimestep(const std::vector<std::string> &);
    void parseNumberOfSteps(const std::vector<std::string> &);

    // parsing general keywords
    void parseStartFilename(const std::vector<std::string> &);
    void parseMoldescriptorFilename(const std::vector<std::string> &);
    void parseGuffPath(const std::vector<std::string> &);
    void parseGuffDatFilename(const std::vector<std::string> &);
    void parseTopologyFilename(const std::vector<std::string> &);
    void parseParameterFilename(const std::vector<std::string> &);

    // parsing output related keywords
    void parseOutputFreq(const std::vector<std::string> &);
    void parseLogFilename(const std::vector<std::string> &);
    void parseInfoFilename(const std::vector<std::string> &);
    void parseEnergyFilename(const std::vector<std::string> &);
    void parseTrajectoryFilename(const std::vector<std::string> &);
    void parseVelocityFilename(const std::vector<std::string> &);
    void parseForceFilename(const std::vector<std::string> &);
    void parseRestartFilename(const std::vector<std::string> &);
    void parseChargeFilename(const std::vector<std::string> &);

    // parsing integrator related keywords
    void parseIntegrator(const std::vector<std::string> &);

    // parsing Box related keywords
    void parseDensity(const std::vector<std::string> &);

    // parsing Virial related keywords TODO: implement
    void parseVirial(const std::vector<std::string> &);

    // parsing simulationBox related keywords
    void parseCoulombRadius(const std::vector<std::string> &);
    void parseCoulombLongRange(const std::vector<std::string> &);
    void parseWolfParameter(const std::vector<std::string> &);

    void parseNonCoulombType(const std::vector<std::string> &);

    // parsing cellList related keywords
    void parseCellListActivated(const std::vector<std::string> &);
    void parseNumberOfCells(const std::vector<std::string> &);

    // parsing temperature related keywords
    void parseThermostat(const std::vector<std::string> &);
    void parseTemperature(const std::vector<std::string> &);
    void parseThermostatRelaxationTime(const std::vector<std::string> &);

    // parsing pressure related keywords
    void parseManostat(const std::vector<std::string> &);
    void parsePressure(const std::vector<std::string> &);
    void parseManostatRelaxationTime(const std::vector<std::string> &);
    void parseCompressibility(const std::vector<std::string> &);

    // parsing reset related keywords
    void parseNScale(const std::vector<std::string> &);
    void parseFScale(const std::vector<std::string> &);
    void parseNReset(const std::vector<std::string> &);
    void parseFReset(const std::vector<std::string> &);

    // parsing constraints related keywords
    void parseShakeActivated(const std::vector<std::string> &);
    void parseShakeTolerance(const std::vector<std::string> &);
    void parseShakeIteration(const std::vector<std::string> &);
    void parseRattleTolerance(const std::vector<std::string> &);
    void parseRattleIteration(const std::vector<std::string> &);

    // parsing information of forcefield
    void parseForceFieldType(const std::vector<std::string> &);

    /********************************
     *                              *
     * standard getters and setters *
     *                              *
     ********************************/

    void setFilename(const std::string_view filename) { _filename = filename; }

    int  getKeywordCount(const std::string &keyword) { return _keywordCountMap[keyword]; }
    void setKeywordCount(const std::string &keyword, const int count) { _keywordCountMap[keyword] = count; }

    bool getKeywordRequired(const std::string &keyword) { return _keywordRequiredMap[keyword]; }
};

#endif   // _INPUT_FILE_READER_HPP_
