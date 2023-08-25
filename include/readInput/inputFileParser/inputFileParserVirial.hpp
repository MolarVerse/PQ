#ifndef _INPUT_FILE_PARSER_VIRIAL_HPP_

#define _INPUT_FILE_PARSER_VIRIAL_HPP_

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
     * @class InputFileParserVirial inherits from InputFileParser
     *
     * @brief Parses the virial commands in the input file
     *
     */
    class InputFileParserVirial : public InputFileParser
    {
      public:
        explicit InputFileParserVirial(engine::Engine &);

        void parseVirial(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_VIRIAL_HPP_