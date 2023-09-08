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

namespace readInput
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

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_HPP_