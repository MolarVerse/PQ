#ifndef _INPUT_FILE_PARSER_CELL_LIST_HPP_

#define _INPUT_FILE_PARSER_CELL_LIST_HPP_

#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput
{
    /**
     * @class InputFileParserCellList inherits from InputFileParser
     *
     * @brief Parses the cell list commands in the input file
     *
     */
    class InputFileParserCellList : public InputFileParser
    {
      public:
        explicit InputFileParserCellList(engine::Engine &);

        void parseCellListActivated(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseNumberOfCells(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_CELL_LIST_HPP_
