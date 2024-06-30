/*****************************************************************************
<GPL_HEADER>

    PQ
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

#ifndef _INPUT_FILE_READER_HPP_

#define _INPUT_FILE_READER_HPP_

#include <cstddef>       // for size_t
#include <functional>    // for function
#include <map>           // for map
#include <memory>        // for unique_ptr
#include <string>        // for string, operator<=>
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for pq::ParseFunc

namespace engine
{
    class Engine;   // forward declaration
}

/**
 * @brief namespace for reading input files
 *
 */
namespace input
{
    void readInputFile(const std::string_view &fileName, engine::Engine &);
    void readJobType(const std::string &fileName, std::unique_ptr<engine::Engine> &);
    void processEqualSign(std::string &command, const size_t lineNumber);

    /**
     * @class InputFileReader
     *
     * @brief reads input file and sets settings
     *
     */
    class InputFileReader
    {
       private:
        std::string     _fileName;
        engine::Engine &_engine;

        std::map<std::string, pq::ParseFunc> _keywordFuncMap;
        std::map<std::string, size_t>        _keywordCountMap;
        std::map<std::string, bool>          _keywordRequiredMap;

        std::vector<std::unique_ptr<InputFileParser>> _parsers;

        size_t _lineNumber = 1;

       public:
        explicit InputFileReader(const std::string_view &, engine::Engine &);

        void read();
        void addKeywords();
        void process(const pq::strings &lineElements);
        void postProcess();

        /***************************
         * standard setter methods *
         ***************************/

        void setFilename(const std::string_view fileName);
        void setKeywordCount(const std::string &keyword, const size_t count);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getKeywordCount(const std::string &keyword);
        [[nodiscard]] bool   getKeywordRequired(const std::string &keyword);

        // clang-format off
        [[nodiscard]] std::map<std::string, size_t> getKeywordCountMap() const;
        [[nodiscard]] std::map<std::string, bool> getKeywordRequiredMap() const;
        [[nodiscard]] std::map<std::string, pq::ParseFunc> getKeywordFuncMap() const;
        // clang-format on
    };

}   // namespace input

#endif   // _INPUT_FILE_READER_HPP_
