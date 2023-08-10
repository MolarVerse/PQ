#ifndef _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

#define _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserNonCoulombType;
}   // namespace readInput

/**
 * @class InputFileParserNonCoulombType inherits from InputFileParser
 *
 * @brief Parses the non-Coulomb type commands in the input file
 *
 */
class readInput::InputFileParserNonCoulombType : public readInput::InputFileParser
{
  public:
    explicit InputFileParserNonCoulombType(engine::Engine &);

    void parseNonCoulombType(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_