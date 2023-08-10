#ifndef _INPUT_FILE_PARSER_INTEGRATOR_HPP_

#define _INPUT_FILE_PARSER_INTEGRATOR_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserIntegrator;
}   // namespace readInput

/**
 * @class InputFileParserIntegrator inherits from InputFileParser
 *
 * @brief Parses the integrator commands in the input file
 *
 */
class readInput::InputFileParserIntegrator : public readInput::InputFileParser
{
  public:
    explicit InputFileParserIntegrator(engine::Engine &);

    void parseIntegrator(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_INTEGRATOR_HPP_