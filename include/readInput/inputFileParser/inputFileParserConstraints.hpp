#ifndef _INPUT_FILE_PARSER_CONSTRAINTS_HPP_

#define _INPUT_FILE_PARSER_CONSTRAINTS_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserConstraints;
}   // namespace readInput

/**
 * @class InputFileParserConstraints inherits from InputFileParser
 *
 * @brief Parses the constraints commands in the input file
 *
 */
class readInput::InputFileParserConstraints : public readInput::InputFileParser
{
  public:
    explicit InputFileParserConstraints(engine::Engine &);

    void parseShakeActivated(const std::vector<std::string> &, const size_t);
    void parseShakeTolerance(const std::vector<std::string> &, const size_t);
    void parseShakeIteration(const std::vector<std::string> &, const size_t);
    void parseRattleTolerance(const std::vector<std::string> &, const size_t);
    void parseRattleIteration(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_CONSTRAINTS_HPP_