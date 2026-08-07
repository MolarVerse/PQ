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

#ifndef _MOLDESCRIPTOR_READER_HPP_

#define _MOLDESCRIPTOR_READER_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string
#include <vector>    // for vector

#include "defaults.hpp"
#include "typeAliases.hpp"

namespace input::molDescriptor
{
    void readMolDescriptor(pq::Engine &);

    /**
     * @class MoldescriptorReader
     *
     * @brief Reads a moldescriptor file
     *
     */
    class MoldescriptorReader
    {
       private:
        int           _lineNumber;
        std::string   _fileName = defaults::_MOLDESCRIPTOR_FILE_DEFAULT_;
        std::ifstream _fp;

        pq::Engine &_engine;

       public:
        explicit MoldescriptorReader(pq::Engine &engine);

        void read();
        void processMolecule(pq::strings &lineElements);
        void convertExternalToInternalAtomTypes(pq::MoleculeType &) const;
    };

}   // namespace input::molDescriptor

#endif   // _MOLDESCRIPTOR_READER_HPP_