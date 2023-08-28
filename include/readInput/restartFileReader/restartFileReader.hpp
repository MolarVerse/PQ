#ifndef _RESTART_FILE_READER_HPP_

#define _RESTART_FILE_READER_HPP_

#include "atomSection.hpp"          // for AtomSection
#include "restartFileSection.hpp"   // for RstFileSection

#include <fstream>   // for ifstream
#include <memory>    // for unique_ptr, make_unique
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput::restartFile
{
    void readRestartFile(engine::Engine &);

    /**
     * @class RestartFileReader
     *
     * @brief Reads a .rst file and returns a SimulationBox object
     *
     */
    class RestartFileReader
    {
      private:
        const std::string _fileName;
        std::ifstream     _fp;
        engine::Engine   &_engine;

        std::unique_ptr<RestartFileSection>              _atomSection = std::make_unique<AtomSection>();
        std::vector<std::unique_ptr<RestartFileSection>> _sections;

      public:
        RestartFileReader(const std::string &, engine::Engine &);

        void                read();
        RestartFileSection *determineSection(std::vector<std::string> &lineElements);
    };

}   // namespace readInput::restartFile

#endif   // _RESTART_FILE_READER_HPP_