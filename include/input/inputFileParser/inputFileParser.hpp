/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _INPUT_FILE_PARSER_HPP_

#define _INPUT_FILE_PARSER_HPP_

#include <cstddef>       // for size_t
#include <functional>    // for function
#include <map>           // for map
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace input
{
    void checkEqualSign(const std::string_view &view, const size_t lineNumber);
    void checkCommand(const std::vector<std::string> &lineElements, const size_t lineNumber);
    void checkCommandArray(const std::vector<std::string> &lineElements, const size_t lineNumber);

    using ParseFunc = std::function<void(const std::vector<std::string> &, const size_t)>;

    /**
     * @class InputFileParser
     *
     * @brief Base class for parsing the input file
     *
     */
    class InputFileParser
    {
      protected:
        engine::Engine &_engine;

        std::map<std::string, ParseFunc> _keywordFuncMap;
        std::map<std::string, bool>      _keywordRequiredMap;
        std::map<std::string, int>       _keywordCountMap;

      public:
        explicit InputFileParser(engine::Engine &engine) : _engine(engine){};

        void                                           addKeyword(const std::string &, ParseFunc, bool);
        [[nodiscard]] std::map<std::string, ParseFunc> getKeywordFuncMap() const { return _keywordFuncMap; }
        [[nodiscard]] std::map<std::string, bool>      getKeywordRequiredMap() const { return _keywordRequiredMap; }
        [[nodiscard]] std::map<std::string, int>       getKeywordCountMap() const { return _keywordCountMap; }
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_HPP_