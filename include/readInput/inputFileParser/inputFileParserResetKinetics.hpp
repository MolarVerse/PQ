#ifndef _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#define _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserResetKinetics;
}   // namespace readInput

/**
 * @class InputFileParserResetKinetics inherits from InputFileParser
 *
 * @brief Parses the reset kinetics commands in the input file
 *
 */
class readInput::InputFileParserResetKinetics : public readInput::InputFileParser
{
  public:
    explicit InputFileParserResetKinetics(engine::Engine &);

    void parseNScale(const std::vector<std::string> &, const size_t);
    void parseFScale(const std::vector<std::string> &, const size_t);
    void parseNReset(const std::vector<std::string> &, const size_t);
    void parseFReset(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_RESET_KINETICS_HPP_