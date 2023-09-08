#ifndef _RESTART_FILE_SECTION_HPP_

#define _RESTART_FILE_SECTION_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string, allocator
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput::restartFile
{
    /**
     * @class RestartFileSection
     *
     * @brief Base class for all sections of a .rst file
     *
     */
    class RestartFileSection
    {
      public:
        virtual ~RestartFileSection() = default;

        int                 _lineNumber;
        std::ifstream      *_fp;
        virtual std::string keyword()                                                         = 0;
        virtual bool        isHeader()                                                        = 0;
        virtual void        process(std::vector<std::string> &lineElements, engine::Engine &) = 0;
    };

}   // namespace readInput::restartFile

#endif   // _RESTART_FILE_SECTION_HPP_