#ifndef _INPUT_FILE_PARSER_TIMINGS_HPP_

#define _INPUT_FILE_PARSER_TIMINGS_HPP_

#include "inputFileParser.hpp"

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
     * @class InputFileParserTimings inherits from InputFileParser
     *
     * @brief Parses the timings commands in the input file
     *
     */
    class InputFileParserTimings : public InputFileParser
    {
      public:
        explicit InputFileParserTimings(engine::Engine &);

        void parseTimeStep(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseNumberOfSteps(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_TIMINGS_HPP_