#ifndef _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

#define _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

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
     * @class InputFileParserSimulationBox inherits from InputFileParser
     *
     * @brief Parses the simulation box commands in the input file
     *
     */
    class InputFileParserSimulationBox : public InputFileParser
    {
      public:
        explicit InputFileParserSimulationBox(engine::Engine &);

        void parseCoulombRadius(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseDensity(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseInitializeVelocities(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_