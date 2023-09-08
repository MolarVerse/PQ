#ifndef _INPUT_FILE_PARSER_OUTPUT_HPP_

#define _INPUT_FILE_PARSER_OUTPUT_HPP_

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
     * @class InputFileParserOutput inherits from InputFileParser
     *
     * @brief Parses the output commands in the input file
     *
     */
    class InputFileParserOutput : public InputFileParser
    {
      public:
        explicit InputFileParserOutput(engine::Engine &);

        void parseOutputFreq(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseLogFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseInfoFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseEnergyFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_OUTPUT_HPP_