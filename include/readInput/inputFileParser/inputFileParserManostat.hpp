#ifndef _INPUT_FILE_PARSER_MANOSTAT_HPP_

#define _INPUT_FILE_PARSER_MANOSTAT_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserManostat;
}   // namespace readInput

/**
 * @class InputFileParserManostat inherits from InputFileParser
 *
 * @brief Parses the manostat commands in the input file
 *
 */
class readInput::InputFileParserManostat : public readInput::InputFileParser
{
  public:
    explicit InputFileParserManostat(engine::Engine &);

    void parseManostat(const std::vector<std::string> &, const size_t);
    void parsePressure(const std::vector<std::string> &, const size_t);
    void parseManostatRelaxationTime(const std::vector<std::string> &, const size_t);
    void parseCompressibility(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_MANOSTAT_HPP_