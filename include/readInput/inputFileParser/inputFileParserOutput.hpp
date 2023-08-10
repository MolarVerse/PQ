#ifndef _INPUT_FILE_PARSER_OUTPUT_HPP_

#define _INPUT_FILE_PARSER_OUTPUT_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserOutput;
}   // namespace readInput

/**
 * @class InputFileParserOutput inherits from InputFileParser
 *
 * @brief Parses the output commands in the input file
 *
 */
class readInput::InputFileParserOutput : public readInput::InputFileParser
{
  public:
    explicit InputFileParserOutput(engine::Engine &);

    void parseOutputFreq(const std::vector<std::string> &, const size_t);
    void parseLogFilename(const std::vector<std::string> &, const size_t);
    void parseInfoFilename(const std::vector<std::string> &, const size_t);
    void parseEnergyFilename(const std::vector<std::string> &, const size_t);
    void parseTrajectoryFilename(const std::vector<std::string> &, const size_t);
    void parseVelocityFilename(const std::vector<std::string> &, const size_t);
    void parseForceFilename(const std::vector<std::string> &, const size_t);
    void parseRestartFilename(const std::vector<std::string> &, const size_t);
    void parseChargeFilename(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_OUTPUT_HPP_