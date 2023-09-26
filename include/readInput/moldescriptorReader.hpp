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

#ifndef _MOLDESCRIPTOR_READER_HPP_

#define _MOLDESCRIPTOR_READER_HPP_

#include "defaults.hpp"

#include <fstream>   // for ifstream
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace simulationBox
{
    class MoleculeType;   // Forward declaration
}

namespace readInput::molDescriptor
{
    void readMolDescriptor(engine::Engine &);

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
        std::string   _fileName = defaults::_MOLDESCRIPTOR_FILENAME_DEFAULT_;
        std::ifstream _fp;

        engine::Engine &_engine;

      public:
        explicit MoldescriptorReader(engine::Engine &engine);

        void read();
        void processMolecule(std::vector<std::string> &lineElements);
        void convertExternalToInternalAtomTypes(simulationBox::MoleculeType &) const;
    };

}   // namespace readInput::molDescriptor

#endif   // _MOLDESCRIPTOR_READER_HPP_