#ifndef _INPUT_FILE_PARSER_HPP_

#define _INPUT_FILE_PARSER_HPP_

#include "engine.hpp"

namespace readInput
{
    class InputFileParser;
    class InputFileParserCellList;
    class InputFileParserConstraints;
    class InputFileParserCoulombLongRange;
    class InputFileParserForceField;
    class InputFileParserGeneral;
    class InputFileParserIntegrator;
    class InputFileParserManostat;
    class InputFileParserNonCoulombType;
    class InputFileParserOutput;
    class InputFileParserParameterFile;
    class InputFileParserResetKinetics;
    class InputFileParserSimulationBox;
    class InputFileParserThermostat;
    class InputFileParserTimings;
    class InputFileParserTopology;
    class InputFileParserVirial;

    void checkEqualSign(const std::string_view &view, const size_t);
    void checkCommand(const std::vector<std::string> &, const size_t);
    void checkCommandArray(const std::vector<std::string> &, const size_t);
}   // namespace readInput

using ParseFunc = std::function<void(const std::vector<std::string> &, const size_t)>;

class readInput::InputFileParser
{
  protected:
    engine::Engine &_engine;

    std::map<std::string, ParseFunc> _keywordFuncMap;
    std::map<std::string, bool>      _keywordRequiredMap;
    std::map<std::string, int>       _keywordCountMap;

  public:
    explicit InputFileParser(engine::Engine &engine) : _engine(engine){};

    void                                           addKeyword(const std::string &, ParseFunc, bool);
    [[nodiscard]] std::map<std::string, ParseFunc> getKeywordFuncMap() const { return _keywordFuncMap; }
    [[nodiscard]] std::map<std::string, bool>      getKeywordRequiredMap() const { return _keywordRequiredMap; }
    [[nodiscard]] std::map<std::string, int>       getKeywordCountMap() const { return _keywordCountMap; }
};

class readInput::InputFileParserCellList : public readInput::InputFileParser
{
  public:
    explicit InputFileParserCellList(engine::Engine &);

    void parseCellListActivated(const std::vector<std::string> &, const size_t);
    void parseNumberOfCells(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserConstraints : public readInput::InputFileParser
{
  public:
    explicit InputFileParserConstraints(engine::Engine &);

    void parseShakeActivated(const std::vector<std::string> &, const size_t);
    void parseShakeTolerance(const std::vector<std::string> &, const size_t);
    void parseShakeIteration(const std::vector<std::string> &, const size_t);
    void parseRattleTolerance(const std::vector<std::string> &, const size_t);
    void parseRattleIteration(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserCoulombLongRange : public readInput::InputFileParser
{
  public:
    explicit InputFileParserCoulombLongRange(engine::Engine &);

    void parseCoulombLongRange(const std::vector<std::string> &, const size_t);
    void parseWolfParameter(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserForceField : public readInput::InputFileParser
{
  public:
    explicit InputFileParserForceField(engine::Engine &);

    void parseForceFieldType(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserGeneral : public readInput::InputFileParser
{
  public:
    explicit InputFileParserGeneral(engine::Engine &);

    void parseStartFilename(const std::vector<std::string> &, const size_t);
    void parseMoldescriptorFilename(const std::vector<std::string> &, const size_t);
    void parseGuffPath(const std::vector<std::string> &, const size_t);
    void parseGuffDatFilename(const std::vector<std::string> &, const size_t);
    void parseJobType(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserIntegrator : public readInput::InputFileParser
{
  public:
    explicit InputFileParserIntegrator(engine::Engine &);

    void parseIntegrator(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserManostat : public readInput::InputFileParser
{
  public:
    explicit InputFileParserManostat(engine::Engine &);

    void parseManostat(const std::vector<std::string> &, const size_t);
    void parsePressure(const std::vector<std::string> &, const size_t);
    void parseManostatRelaxationTime(const std::vector<std::string> &, const size_t);
    void parseCompressibility(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserNonCoulombType : public readInput::InputFileParser
{
  public:
    explicit InputFileParserNonCoulombType(engine::Engine &);

    void parseNonCoulombType(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserOutput : public readInput::InputFileParser
{
  public:
    explicit InputFileParserOutput(engine::Engine &);

    void parseOutputFreq(const std::vector<std::string> &, const size_t);
    void parseLogFilename(const std::vector<std::string> &, const size_t);
    void parseInfoFilename(const std::vector<std::string> &, const size_t);
    void parseEnergyFilename(const std::vector<std::string> &, const size_t);
    void parseTrajectoryFilename(const std::vector<std::string> &, const size_t);
    void parseVelocityFilename(const std::vector<std::string> &, const size_t);
    void parseForceFilename(const std::vector<std::string> &, const size_t);
    void parseRestartFilename(const std::vector<std::string> &, const size_t);
    void parseChargeFilename(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserParameterFile : public readInput::InputFileParser
{
  public:
    explicit InputFileParserParameterFile(engine::Engine &);

    void parseParameterFilename(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserResetKinetics : public readInput::InputFileParser
{
  public:
    explicit InputFileParserResetKinetics(engine::Engine &);

    void parseNScale(const std::vector<std::string> &, const size_t);
    void parseFScale(const std::vector<std::string> &, const size_t);
    void parseNReset(const std::vector<std::string> &, const size_t);
    void parseFReset(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserSimulationBox : public readInput::InputFileParser
{
  public:
    explicit InputFileParserSimulationBox(engine::Engine &);

    void parseCoulombRadius(const std::vector<std::string> &, const size_t);
    void parseDensity(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserThermostat : public readInput::InputFileParser
{
  public:
    explicit InputFileParserThermostat(engine::Engine &);

    void parseThermostat(const std::vector<std::string> &, const size_t);
    void parseTemperature(const std::vector<std::string> &, const size_t);
    void parseThermostatRelaxationTime(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserTimings : public readInput::InputFileParser
{
  public:
    explicit InputFileParserTimings(engine::Engine &);

    void parseTimeStep(const std::vector<std::string> &, const size_t);
    void parseNumberOfSteps(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserTopology : public readInput::InputFileParser
{
  public:
    explicit InputFileParserTopology(engine::Engine &);

    void parseTopologyFilename(const std::vector<std::string> &, const size_t);
};

class readInput::InputFileParserVirial : public readInput::InputFileParser
{
  public:
    explicit InputFileParserVirial(engine::Engine &);

    void parseVirial(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_HPP_