#ifndef _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#define _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#include "inputFileParser.hpp"

#include <cstddef>   // for size_t
#include <string>
#include <vector>

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    /**
     * @class InputFileParserResetKinetics inherits from InputFileParser
     *
     * @brief Parses the reset kinetics commands in the input file
     *
     */
    class InputFileParserResetKinetics : public InputFileParser
    {
      public:
        explicit InputFileParserResetKinetics(engine::Engine &);

        void parseNScale(const std::vector<std::string> &, const size_t);
        void parseFScale(const std::vector<std::string> &, const size_t);
        void parseNReset(const std::vector<std::string> &, const size_t);
        void parseFReset(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_RESET_KINETICS_HPP_