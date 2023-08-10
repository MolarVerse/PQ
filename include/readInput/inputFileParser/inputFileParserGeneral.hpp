#ifndef _INPUT_FILE_PARSER_GENERAL_HPP_

#define _INPUT_FILE_PARSER_GENERAL_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserGeneral;
}   // namespace readInput

/**
 * @class InputFileParserGeneral inherits from InputFileParser
 *
 * @brief Parses the general commands in the input file
 *
 */
class readInput::InputFileParserGeneral : public readInput::InputFileParser
{
  public:
    explicit InputFileParserGeneral(engine::Engine &);

    void parseStartFilename(const std::vector<std::string> &, const size_t);
    void parseMoldescriptorFilename(const std::vector<std::string> &, const size_t);
    void parseGuffDatFilename(const std::vector<std::string> &, const size_t);
    void parseJobType(const std::vector<std::string> &, const size_t);

    [[noreturn]] void parseGuffPath(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_GENERAL_HPP_