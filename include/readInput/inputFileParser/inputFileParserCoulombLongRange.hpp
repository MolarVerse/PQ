#ifndef _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_

#define _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserCoulombLongRange;
}   // namespace readInput

/**
 * @class InputFileParserCoulombLongRange inherits from InputFileParser
 *
 * @brief Parses the Coulomb long range commands in the input file
 *
 */
class readInput::InputFileParserCoulombLongRange : public readInput::InputFileParser
{
  public:
    explicit InputFileParserCoulombLongRange(engine::Engine &);

    void parseCoulombLongRange(const std::vector<std::string> &, const size_t);
    void parseWolfParameter(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_COULOMB_LONG_RANGE_HPP_