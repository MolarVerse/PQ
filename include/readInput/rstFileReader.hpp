#ifndef _RST_FILE_READER_HPP_

#define _RST_FILE_READER_HPP_

#include "rstFileSection.hpp"   // for AtomSection, RstFileSection

#include <fstream>   // for ifstream
#include <memory>    // for unique_ptr, make_unique
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    void readRstFile(engine::Engine &);

    /**
     * @class RstFileReader
     *
     * @brief Reads a .rst file and returns a SimulationBox object
     *
     */
    class RstFileReader
    {
      private:
        const std::string _filename;
        std::ifstream     _fp;
        engine::Engine   &_engine;

        std::unique_ptr<RstFileSection>              _atomSection = std::make_unique<AtomSection>();
        std::vector<std::unique_ptr<RstFileSection>> _sections;

      public:
        RstFileReader(const std::string &, engine::Engine &);

        void            read();
        RstFileSection *determineSection(std::vector<std::string> &);
    };

}   // namespace readInput

#endif   // _RST_FILE_READER_HPP_