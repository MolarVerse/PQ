#ifndef _INPUT_FILE_PARSER_MANOSTAT_HPP_

#define _INPUT_FILE_PARSER_MANOSTAT_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <stddef.h>   // for size_t
#include <string>     // for string
#include <vector>     // for vector

namespace engine
{
    class Engine;
}   // namespace engine

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

        void parseManostat(const std::vector<std::string> &, const size_t);
        void parsePressure(const std::vector<std::string> &, const size_t);
        void parseManostatRelaxationTime(const std::vector<std::string> &, const size_t);
        void parseCompressibility(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_MANOSTAT_HPP_