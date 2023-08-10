#ifndef _INPUT_FILE_PARSER_VIRIAL_HPP_

#define _INPUT_FILE_PARSER_VIRIAL_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserVirial;
}   // namespace readInput

/**
 * @class InputFileParserVirial inherits from InputFileParser
 *
 * @brief Parses the virial commands in the input file
 *
 */
class readInput::InputFileParserVirial : public readInput::InputFileParser
{
  public:
    explicit InputFileParserVirial(engine::Engine &);

    void parseVirial(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_VIRIAL_HPP_