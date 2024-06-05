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

#ifndef _M_SHAKE_READER_HPP_

#define _M_SHAKE_READER_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string

namespace engine
{
    class Engine;   // Forward declaration
}

namespace input::mShake
{
    void readMShake(engine::Engine &engine);

    /**
     * @class MShakeReader
     *
     * @brief Reads a mShake file
     *
     */
    class MShakeReader
    {
       private:
        int           _lineNumber;
        std::string   _fileName;
        std::ifstream _fp;

        engine::Engine &_engine;

       public:
        explicit MShakeReader(engine::Engine &engine);

        void read();
        void processCommentLine(std::string &line);
        void processAtomLine(std::string &line);
    };
}   // namespace input::mShake

#endif   // _M_SHAKE_READER_HPP_