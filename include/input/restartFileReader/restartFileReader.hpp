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

#ifndef _RESTART_FILE_READER_HPP_

#define _RESTART_FILE_READER_HPP_

#include <fstream>   // for ifstream
#include <memory>    // for unique_ptr, make_unique
#include <string>    // for string
#include <vector>    // for vector

#include "atomSection.hpp"          // for AtomSection
#include "restartFileSection.hpp"   // for RstFileSection
#include "typeAliases.hpp"

namespace input::restartFile
{
    void readRestartFile(pq::Engine &);

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
        pq::Engine       &_engine;

        pq::UniqueRestartSection _atomSection = std::make_unique<AtomSection>();
        pq::UniqueRestartSectionVec _sections;

       public:
        RestartFileReader(const std::string &, pq::Engine &);

        void                read();
        RestartFileSection *determineSection(pq::strings &lineElements);
    };

}   // namespace input::restartFile

#endif   // _RESTART_FILE_READER_HPP_