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

namespace input::parameterFile
{
    using section_unique_ptr = std::unique_ptr<ParameterFileSection>;
    void               readParameterFile(engine::Engine &);
    [[nodiscard]] bool isNeeded();

    /**
     * @class ParameterReader
     *
     * @brief reads parameter file and sets settings
     *
     */
    class ParameterFileReader
    {
      private:
        std::string     _fileName;
        std::ifstream   _fp;
        engine::Engine &_engine;

        std::vector<std::unique_ptr<ParameterFileSection>> _parameterFileSections;

      public:
        ParameterFileReader(const std::string &filename, engine::Engine &engine);

        void read();

        [[nodiscard]] ParameterFileSection *determineSection(const std::vector<std::string> &lineElements);
        void                                deleteSection(const ParameterFileSection *section);

        /**************************************
         * standard getter and setter methods *
         **************************************/

        void setFilename(const std::string_view &filename) { _fileName = filename; }

        [[nodiscard]] std::vector<section_unique_ptr> &getParameterFileSections() { return _parameterFileSections; }
        [[nodiscard]] const std::string               &getFilename() const { return _fileName; }
    };

}   // namespace input::parameterFile

#endif   // _PARAMETER_FILE_READER_HPP_