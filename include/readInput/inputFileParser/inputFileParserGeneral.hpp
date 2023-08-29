#ifndef _INPUT_FILE_PARSER_GENERAL_HPP_

#define _INPUT_FILE_PARSER_GENERAL_HPP_

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
     * @class InputFileParserGeneral inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class InputFileParserGeneral : public InputFileParser
    {
      public:
        explicit InputFileParserGeneral(engine::Engine &);

        void parseJobType(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_GENERAL_HPP_