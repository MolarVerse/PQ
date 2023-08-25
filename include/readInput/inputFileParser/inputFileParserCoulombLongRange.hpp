#ifndef _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_

#define _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_

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
     * @class InputFileParserCoulombLongRange inherits from InputFileParser
     *
     * @brief Parses the Coulomb long range commands in the input file
     *
     */
    class InputFileParserCoulombLongRange : public InputFileParser
    {
      public:
        explicit InputFileParserCoulombLongRange(engine::Engine &);

        void parseCoulombLongRange(const std::vector<std::string> &, const size_t);
        void parseWolfParameter(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_