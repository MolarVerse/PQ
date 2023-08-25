#ifndef _INPUT_FILE_PARSER_TOPOLOGY_HPP_

#define _INPUT_FILE_PARSER_TOPOLOGY_HPP_

#include "inputFileParser.hpp"

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    /**
     * @class InputFileParserTopology inherits from InputFileParser
     *
     * @brief Parses the topology commands in the input file
     *
     */
    class InputFileParserTopology : public InputFileParser
    {
      public:
        explicit InputFileParserTopology(engine::Engine &);

        void parseTopologyFilename(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_TOPOLOGY_HPP_