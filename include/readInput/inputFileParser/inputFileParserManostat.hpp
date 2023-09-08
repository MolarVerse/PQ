#ifndef _INPUT_FILE_PARSER_MANOSTAT_HPP_

#define _INPUT_FILE_PARSER_MANOSTAT_HPP_

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
     * @class InputFileParserManostat inherits from InputFileParser
     *
     * @brief Parses the manostat commands in the input file
     *
     */
    class InputFileParserManostat : public InputFileParser
    {
      public:
        explicit InputFileParserManostat(engine::Engine &);

        void parseManostat(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parsePressure(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseManostatRelaxationTime(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseCompressibility(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_MANOSTAT_HPP_