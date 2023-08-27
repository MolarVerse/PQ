#ifndef _PARAMETER_FILE_SECTION_HPP_

#define _PARAMETER_FILE_SECTION_HPP_

#include <cstddef>   // for size_t
#include <iosfwd>    // for ifstream
#include <string>    // for string, allocator
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace potential
{
    enum class NonCoulombType : size_t;   // Forward declaration
}

namespace readInput::parameterFile
{
    /**
     * @class ParameterFileSection
     *
     * @brief base class for reading parameter file sections
     *
     */
    class ParameterFileSection
    {
      protected:
        int            _lineNumber;
        std::ifstream *_fp;

      public:
        virtual ~ParameterFileSection() = default;

        virtual void process(std::vector<std::string> &lineElements, engine::Engine &);
        void         endedNormally(bool);

        virtual std::string keyword()                                                                = 0;
        virtual void        processSection(std::vector<std::string> &lineElements, engine::Engine &) = 0;
        virtual void        processHeader(std::vector<std::string> &lineElements, engine::Engine &)  = 0;

        void setLineNumber(int lineNumber) { _lineNumber = lineNumber; }
        void setFp(std::ifstream *fp) { _fp = fp; }

        [[nodiscard]] int getLineNumber() const { return _lineNumber; }
    };

}   // namespace readInput::parameterFile

#endif   // _PARAMETER_FILE_SECTION_HPP_