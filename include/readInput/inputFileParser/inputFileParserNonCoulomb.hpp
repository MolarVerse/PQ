#ifndef _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

#define _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserNonCoulomb;
}   // namespace readInput

/**
 * @class InputFileParserNonCoulomb inherits from InputFileParser
 *
 * @brief Parses the non-Coulomb type commands in the input file
 *
 */
class readInput::InputFileParserNonCoulomb : public readInput::InputFileParser
{
  public:
    explicit InputFileParserNonCoulomb(engine::Engine &);

    void parseNonCoulombType(const std::vector<std::string> &, const size_t);
    void parseIntraNonBondedFile(const std::vector<std::string> &, const size_t);
    void parseIntraNonBondedType(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_NON_COULOMB_TYPE_HPP_