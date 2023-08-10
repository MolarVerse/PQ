#ifndef _INPUT_FILE_PARSER_TIMINGS_HPP_

#define _INPUT_FILE_PARSER_TIMINGS_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserTimings;
}   // namespace readInput

/**
 * @class InputFileParserTimings inherits from InputFileParser
 *
 * @brief Parses the timings commands in the input file
 *
 */
class readInput::InputFileParserTimings : public readInput::InputFileParser
{
  public:
    explicit InputFileParserTimings(engine::Engine &);

    void parseTimeStep(const std::vector<std::string> &, const size_t);
    void parseNumberOfSteps(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_TIMINGS_HPP_