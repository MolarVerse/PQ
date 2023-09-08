#ifndef _INPUT_FILE_READER_HPP_

#define _INPUT_FILE_READER_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>       // for size_t
#include <functional>    // for function
#include <map>           // for map
#include <memory>        // for unique_ptr
#include <string>        // for string, operator<=>
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

/**
 * @brief namespace for reading input files
 *
 */
namespace readInput
{
    void readInputFile(const std::string_view &fileName, engine::Engine &);

    using ParseFunc = std::function<void(const std::vector<std::string> &lineElements, const size_t lineNumber)>;

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

        std::map<std::string, ParseFunc> _keywordFuncMap;
        std::map<std::string, size_t>    _keywordCountMap;
        std::map<std::string, bool>      _keywordRequiredMap;

        std::vector<std::unique_ptr<InputFileParser>> _parsers;

        size_t _lineNumber = 1;

      public:
        InputFileReader(const std::string_view &fileName, engine::Engine &engine);

        void read();
        void addKeywords();
        void process(const std::vector<std::string> &lineElements);
        void postProcess();

        /********************************
         *                              *
         * standard getters and setters *
         *                              *
         ********************************/

        void setFilename(const std::string_view fileName) { _fileName = fileName; }
        void setKeywordCount(const std::string &keyword, const size_t count) { _keywordCountMap[keyword] = count; }

        [[nodiscard]] size_t getKeywordCount(const std::string &keyword) { return _keywordCountMap[keyword]; }
        [[nodiscard]] bool   getKeywordRequired(const std::string &keyword) { return _keywordRequiredMap[keyword]; }

        [[nodiscard]] std::map<std::string, size_t>    getKeywordCountMap() const { return _keywordCountMap; }
        [[nodiscard]] std::map<std::string, bool>      getKeywordRequiredMap() const { return _keywordRequiredMap; }
        [[nodiscard]] std::map<std::string, ParseFunc> getKeywordFuncMap() const { return _keywordFuncMap; }
    };

}   // namespace readInput

#endif   // _INPUT_FILE_READER_HPP_
