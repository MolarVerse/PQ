#ifndef _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#define _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    /**
     * @class InputFileParserThermostat inherits from InputFileParser
     *
     * @brief Parses the thermostat commands in the input file
     *
     */
    class InputFileParserThermostat : public InputFileParser
    {
      public:
        explicit InputFileParserThermostat(engine::Engine &);

        void parseThermostat(const std::vector<std::string> &, const size_t);
        void parseTemperature(const std::vector<std::string> &, const size_t);
        void parseThermostatRelaxationTime(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_THERMOSTAT_HPP_