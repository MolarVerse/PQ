#ifndef _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_

#define _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserParameterFile;
}   // namespace readInput

/**
 * @class InputFileParserParameterFile inherits from InputFileParser
 *
 * @brief Parses the parameter file commands in the input file
 *
 */
class readInput::InputFileParserParameterFile : public readInput::InputFileParser
{
  public:
    explicit InputFileParserParameterFile(engine::Engine &);

    void parseParameterFilename(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_