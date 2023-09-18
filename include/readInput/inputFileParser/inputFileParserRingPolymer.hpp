#ifndef _INPUT_FILE_PARSER_RING_POLYMER_HPP_

#define _INPUT_FILE_PARSER_RING_POLYMER_HPP_

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
     * @class InputFileParserRingPolymer inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class InputFileParserRingPolymer : public InputFileParser
    {
      public:
        explicit InputFileParserRingPolymer(engine::Engine &);

        void parseNumberOfBeads(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_RING_POLYMER_HPP_