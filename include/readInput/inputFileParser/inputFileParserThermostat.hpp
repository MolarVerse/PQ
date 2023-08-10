#ifndef _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#define _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserThermostat;
}   // namespace readInput

/**
 * @class InputFileParserThermostat inherits from InputFileParser
 *
 * @brief Parses the thermostat commands in the input file
 *
 */
class readInput::InputFileParserThermostat : public readInput::InputFileParser
{
  public:
    explicit InputFileParserThermostat(engine::Engine &);

    void parseThermostat(const std::vector<std::string> &, const size_t);
    void parseTemperature(const std::vector<std::string> &, const size_t);
    void parseThermostatRelaxationTime(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_THERMOSTAT_HPP_