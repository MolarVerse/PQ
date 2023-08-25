#ifndef _INPUT_FILE_PARSER_CONSTRAINTS_HPP_

#define _INPUT_FILE_PARSER_CONSTRAINTS_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

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
     * @class InputFileParserConstraints inherits from InputFileParser
     *
     * @brief Parses the constraints commands in the input file
     *
     */
    class InputFileParserConstraints : public InputFileParser
    {
      public:
        explicit InputFileParserConstraints(engine::Engine &);

        void parseShakeActivated(const std::vector<std::string> &, const size_t);
        void parseShakeTolerance(const std::vector<std::string> &, const size_t);
        void parseShakeIteration(const std::vector<std::string> &, const size_t);
        void parseRattleTolerance(const std::vector<std::string> &, const size_t);
        void parseRattleIteration(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_CONSTRAINTS_HPP_