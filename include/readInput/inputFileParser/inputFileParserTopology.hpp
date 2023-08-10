#ifndef _INPUT_FILE_PARSER_TOPOLOGY_HPP_

#define _INPUT_FILE_PARSER_TOPOLOGY_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserTopology;
}   // namespace readInput

/**
 * @class InputFileParserTopology inherits from InputFileParser
 *
 * @brief Parses the topology commands in the input file
 *
 */
class readInput::InputFileParserTopology : public readInput::InputFileParser
{
  public:
    explicit InputFileParserTopology(engine::Engine &);

    void parseTopologyFilename(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_TOPOLOGY_HPP_