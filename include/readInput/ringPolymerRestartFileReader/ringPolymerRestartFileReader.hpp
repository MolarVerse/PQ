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

#ifndef _RING_POLYMER_RESTART_FILE_READER_HPP_

#define _RING_POLYMER_RESTART_FILE_READER_HPP_

#include <fstream>   // IWYU pragma: keep
#include <string>    // for string

namespace engine
{
    class RingPolymerEngine;   // forward declaration
}

namespace readInput::ringPolymer
{
    void readRingPolymerRestartFile(engine::RingPolymerEngine &);

    /**
     * @class RingPolymerRestartFileReader
     *
     * @brief Reads a .rpmd.rst file sets the ring polymer beads in the engine
     *
     */
    class RingPolymerRestartFileReader
    {
      private:
        const std::string          _fileName;
        std::ifstream              _fp;
        engine::RingPolymerEngine &_engine;

      public:
        RingPolymerRestartFileReader(const std::string &fileName, engine::RingPolymerEngine &engine)
            : _fileName(fileName), _fp(fileName), _engine(engine){};

        void read();
    };
}   // namespace readInput::ringPolymer

#endif   // _RING_POLYMER_RESTART_FILE_READER_HPP_