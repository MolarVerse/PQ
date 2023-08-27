#ifndef _PARAMETER_FILE_READER_HPP_

#define _PARAMETER_FILE_READER_HPP_

#include "parameterFileSection.hpp"

#include <fstream>   // for ifstream
#include <memory>    // for unique_ptr
#include <string>
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput::parameterFile
{
    void readParameterFile(engine::Engine &);

    /**
     * @class ParameterReader
     *
     * @brief reads parameter file and sets settings
     *
     */
    class ParameterFileReader
    {
      private:
        std::string     _filename;
        std::ifstream   _fp;
        engine::Engine &_engine;

        std::vector<std::unique_ptr<ParameterFileSection>> _parameterFileSections;

      public:
        ParameterFileReader(const std::string &filename, engine::Engine &engine);

        bool isNeeded() const;
        void read();

        [[nodiscard]] ParameterFileSection *determineSection(const std::vector<std::string> &lineElements);
        void                                deleteSection(const ParameterFileSection *section);

        void setFilename(const std::string_view &filename) { _filename = filename; }

        [[nodiscard]] std::vector<std::unique_ptr<ParameterFileSection>> &getParameterFileSections()
        {
            return _parameterFileSections;
        }
        [[nodiscard]] const std::string &getFilename() const { return _filename; }
    };

}   // namespace readInput::parameterFile

#endif   // _PARAMETER_FILE_READER_HPP_