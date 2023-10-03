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

namespace input::restartFile
{
    void readRestartFile(engine::Engine &);

    /**
     * @class RestartFileReader
     *
     * @brief Reads a .rst file and sets the simulation box in the engine
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

}   // namespace input::restartFile

#endif   // _RESTART_FILE_READER_HPP_