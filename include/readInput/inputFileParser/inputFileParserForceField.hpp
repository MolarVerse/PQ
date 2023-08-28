#ifndef _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

#define _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

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
     * @class InputFileParserForceField inherits from InputFileParser
     *
     * @brief Parses the force field commands in the input file
     *
     */
    class InputFileParserForceField : public InputFileParser
    {
      public:
        explicit InputFileParserForceField(engine::Engine &);

        void parseForceFieldType(const std::vector<std::string> &lineElements, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_FORCE_FIELD_HPP_