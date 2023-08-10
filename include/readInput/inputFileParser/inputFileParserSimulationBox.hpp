#ifndef _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

#define _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserSimulationBox;
}   // namespace readInput

/**
 * @class InputFileParserSimulationBox inherits from InputFileParser
 *
 * @brief Parses the simulation box commands in the input file
 *
 */
class readInput::InputFileParserSimulationBox : public readInput::InputFileParser
{
  public:
    explicit InputFileParserSimulationBox(engine::Engine &);

    void parseCoulombRadius(const std::vector<std::string> &, const size_t);
    void parseDensity(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_