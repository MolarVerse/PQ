#ifndef _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

#define _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserForceField;
}   // namespace readInput

/**
 * @class InputFileParserForceField inherits from InputFileParser
 *
 * @brief Parses the force field commands in the input file
 *
 */
class readInput::InputFileParserForceField : public readInput::InputFileParser
{
  public:
    explicit InputFileParserForceField(engine::Engine &);

    void parseForceFieldType(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_FORCE_FIELD_HPP_