#ifndef _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

#define _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

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
     * @class InputFileParserNonCoulomb inherits from InputFileParser
     *
     * @brief Parses the non-Coulomb type commands in the input file
     *
     */
    class InputFileParserNonCoulomb : public InputFileParser
    {
      public:
        explicit InputFileParserNonCoulomb(engine::Engine &);

        void parseNonCoulombType(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_