#ifndef _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#define _INPUT_FILE_PARSER_THERMOSTAT_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

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

        void parseThermostat(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseTemperature(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseThermostatRelaxationTime(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseThermostatFriction(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseThermostatCouplingFrequency(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_THERMOSTAT_HPP_